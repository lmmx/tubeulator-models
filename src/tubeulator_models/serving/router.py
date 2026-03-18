"""Public API for tube routing inference."""

from __future__ import annotations

from dataclasses import dataclass, field

from .infer import (
    LoadedModel,
    _assign_lines,
    _compute_cumulative_times,
    _display_name,
    load_model,
    rollout,
    rollout_via,
)


@dataclass
class RouteStep:
    station: str
    line: str | None = None
    cumulative_minutes: float = 0.0
    transfer_minutes: float = 0.0
    is_transfer: bool = False


@dataclass
class Route:
    steps: list[RouteStep] = field(default_factory=list)
    success: bool = False
    total_minutes: float = 0.0
    lines_used: list[str] = field(default_factory=list)
    n_transfers: int = 0

    def __str__(self) -> str:
        if not self.steps:
            return "Empty route"
        parts = [f"{self.steps[0].station}"]
        for s in self.steps[1:]:
            line_tag = f"[{s.line}] " if s.line else ""
            prefix = f"  ↳ {line_tag}" if s.is_transfer else f"  → {line_tag}"
            parts.append(f"{prefix}{s.station} ({s.cumulative_minutes:.1f}m)")
        status = "✓" if self.success else "✗"
        t = "transfer" if self.n_transfers == 1 else "transfers"
        parts.append(
            f"{status} {len(self.steps) - 1} hops · "
            f"{len(self.lines_used)} lines · "
            f"{self.n_transfers} {t} · "
            f"{self.total_minutes:.1f} min"
        )
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "total_minutes": self.total_minutes,
            "lines_used": self.lines_used,
            "n_transfers": self.n_transfers,
            "steps": [
                {
                    "station": s.station,
                    "line": s.line,
                    "cumulative_minutes": s.cumulative_minutes,
                    "transfer_minutes": s.transfer_minutes,
                    "is_transfer": s.is_transfer,
                }
                for s in self.steps
            ],
        }


def _clean_name(name: str) -> str:
    for suffix in (" Underground Station", " DLR Station", " Station"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


class TubeRouter:
    """
    Learned next-hop router for the London Underground.

    Usage::

        from tubeulator_models import TubeRouter

        router = TubeRouter.from_pretrained("permutans/tube-nexthop-policy")
        route = router.route("Plaistow", "Shoreditch")
        print(route)
    """

    def __init__(self, lm: LoadedModel) -> None:
        self._lm = lm

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        *,
        profile: str = "full",
    ) -> TubeRouter:
        """
        Load from a HuggingFace repo ID or local export directory.

        Args:
            source: HF repo (e.g. "permutans/tube-nexthop-policy") or path.
            profile: Config profile for checkpoint loading fallback.
        """
        lm = load_model("policy", source=source, profile=profile)
        return cls(lm)

    @property
    def stations(self) -> list[str]:
        """All station display names."""
        return [self._lm.stop_names.get(sid, sid) for sid in self._lm.stations]

    def route(
        self,
        origin: str,
        destination: str,
        *,
        via: list[str] | None = None,
        max_hops: int = 60,
    ) -> Route:
        """
        Compute a route from origin to destination.

        Args:
            origin: Station name (substring match supported).
            destination: Station name.
            via: Optional waypoints to route through in order.
            max_hops: Safety limit per segment.

        Returns:
            Route object with steps, timings, and line assignments.
        """
        lm = self._lm

        if via:
            path, success = rollout_via(lm, origin, destination, via, max_hops=max_hops)
        else:
            path, success = rollout(lm, origin, destination, max_hops=max_hops)

        steps: list[RouteStep] = []
        has_topo = lm.topo is not None and len(path) >= 2

        if has_topo:
            segments = _assign_lines(path, lm.stations, lm.topo)
            cum, _est, xfer = _compute_cumulative_times(
                path, segments, lm.stations, lm.topo, lm.transfer_lookup
            )
            prev_line = None
            for i, idx in enumerate(path):
                name = _clean_name(_display_name(idx, lm.stations, lm.stop_names))
                line = segments[i][0] if i < len(segments) else None
                is_xfer = (
                    prev_line is not None and line is not None and line != prev_line
                )

                # Skip hub-internal hops (no line on this segment)
                # unless it's the final destination
                if line is None and i < len(path) - 1:
                    continue

                steps.append(
                    RouteStep(
                        station=name,
                        line=line,
                        cumulative_minutes=cum[i] / 60.0,
                        transfer_minutes=xfer[i] / 60.0 if i < len(xfer) else 0.0,
                        is_transfer=is_xfer,
                    )
                )
                if i < len(segments) and segments[i][0] is not None:
                    prev_line = segments[i][0]
        else:
            for i, idx in enumerate(path):
                name = _clean_name(_display_name(idx, lm.stations, lm.stop_names))
                steps.append(RouteStep(station=name))

        lines_used = []
        seen: set[str] = set()
        if has_topo:
            for s in steps:
                if s.line and s.line not in seen:
                    lines_used.append(s.line)
                    seen.add(s.line)

        n_transfers = sum(1 for s in steps if s.is_transfer)
        total = steps[-1].cumulative_minutes if steps else 0.0

        return Route(
            steps=steps,
            success=success,
            total_minutes=total,
            lines_used=lines_used,
            n_transfers=n_transfers,
        )

    def routes(
        self,
        origin: str,
        destination: str,
        *,
        n: int = 3,
        beam_width: int = 8,
        max_time_ratio: float = 2.0,
        max_hops: int = 60,
    ) -> list[Route]:
        """
        Compute multiple diverse routes from origin to destination.

        Args:
            origin: Station name (substring match supported).
            destination: Station name.
            n: Maximum number of distinct routes to return.
            beam_width: Beam search width (higher = more candidates before dedup).
            max_time_ratio: Drop routes slower than this × the fastest route.
            max_hops: Safety limit per rollout.

        Returns:
            List of Route objects, best-first.
        """
        from .infer import (
            _assign_lines,
            _compute_cumulative_times,
            rollout_diverse,
        )

        lm = self._lm
        candidates = rollout_diverse(
            lm,
            origin,
            destination,
            n_routes=n,
            beam_width=beam_width,
            max_time_ratio=max_time_ratio,
            max_hops=max_hops,
        )

        results = []
        for path, score in candidates:
            steps: list[RouteStep] = []
            has_topo = lm.topo is not None and len(path) >= 2

            if has_topo:
                segments = _assign_lines(path, lm.stations, lm.topo)
                cum, _est, xfer = _compute_cumulative_times(
                    path, segments, lm.stations, lm.topo, lm.transfer_lookup
                )
                prev_line = None
                for i, idx in enumerate(path):
                    name = _clean_name(_display_name(idx, lm.stations, lm.stop_names))
                    line = segments[i][0] if i < len(segments) else None
                    is_xfer = (
                        prev_line is not None and line is not None and line != prev_line
                    )

                    # Skip hub-internal hops (no line on this segment)
                    # unless it's the final destination
                    if line is None and i < len(path) - 1:
                        continue

                    steps.append(
                        RouteStep(
                            station=name,
                            line=line,
                            cumulative_minutes=cum[i] / 60.0,
                            transfer_minutes=xfer[i] / 60.0 if i < len(xfer) else 0.0,
                            is_transfer=is_xfer,
                        )
                    )
                    if i < len(segments) and segments[i][0] is not None:
                        prev_line = segments[i][0]
            else:
                for i, idx in enumerate(path):
                    name = _clean_name(_display_name(idx, lm.stations, lm.stop_names))
                    steps.append(RouteStep(station=name))

            lines_used = []
            seen: set[str] = set()
            if has_topo:
                for s in steps:
                    if s.line and s.line not in seen:
                        lines_used.append(s.line)
                        seen.add(s.line)

            n_transfers = sum(1 for s in steps if s.is_transfer)
            total = steps[-1].cumulative_minutes if steps else 0.0

            results.append(
                Route(
                    steps=steps,
                    success=True,
                    total_minutes=total,
                    lines_used=lines_used,
                    n_transfers=n_transfers,
                )
            )

        return results
