"""Training loop for route-prediction models."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .beam import beam_decode, rollout_nexthop
from .config import TrainConfig
from .dataset import PAD, GPURouteDataset, NextHopGPUDataset
from .defaults import MODEL_TYPES
from .evaluate import (
    NextHopMetrics,
    RouteMetrics,
    compute_metrics,
    compute_nexthop_rollout_metrics,
    compute_nexthop_step_metrics,
)
from .graph_enriched import build_enriched_graph
from .metrics_log import MetricsLogger
from .models.combined import RouteModel
from .topology import build_adj_mask, build_line_station_mask, extract


__all__ = ["train"]


def _compute_min_route_loss(
    logits: dict,
    all_valid_labels: list[list[torch.Tensor]],
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """CE loss against whichever valid route fits best — single kernel call."""
    station_logits = logits["station"]  # (B, T, V)
    B, T, V = station_logits.shape
    device = station_logits.device

    max_routes = max(len(r) for r in all_valid_labels)

    # Build (B, R, T) label tensor — one fused allocation
    padded = torch.full((B, max_routes, T), PAD, dtype=torch.long, device=device)
    route_valid = torch.zeros(B, max_routes, dtype=torch.bool, device=device)

    for b in range(B):
        for r, label in enumerate(all_valid_labels[b]):
            L = min(label.size(0), T)
            padded[b, r, :L] = label[:L]
            route_valid[b, r] = True

    # Single CE call over all (B × R × T) tokens
    logits_exp = station_logits.unsqueeze(1).expand(-1, max_routes, -1, -1)
    ce = nn.CrossEntropyLoss(
        ignore_index=PAD, reduction="none", label_smoothing=label_smoothing
    )
    per_token = ce(
        logits_exp.reshape(-1, V),
        padded.reshape(-1),
    ).view(B, max_routes, T)

    # Mean over valid tokens per route, then min over routes
    n_tokens = (padded != PAD).sum(dim=-1).float().clamp(min=1)  # (B, R)
    route_loss = per_token.sum(dim=-1) / n_tokens  # (B, R)
    route_loss[~route_valid] = float("inf")

    return route_loss.min(dim=1).values.mean()


def _compute_loss(
    logits: dict,
    labels: torch.Tensor,
    model_type: str,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Vectorised loss — one CE call per head, not one per step×head."""
    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=label_smoothing)

    if model_type in ("line", "change"):
        keys = ["line", "dir"] if model_type == "line" else ["line", "dir", "station"]
        s = len(keys)  # stride: 2 for line, 3 for change
        max_legs = logits["line"].size(1)

        # Pad labels to at least max_legs * stride columns
        needed = max_legs * s
        if labels.size(1) < needed:
            pad = labels.new_full((labels.size(0), needed - labels.size(1)), PAD)
            labels = torch.cat([labels, pad], dim=1)

        loss = torch.tensor(0.0, device=labels.device)
        for offset, key in enumerate(keys):
            # Extract every s-th column starting at offset → (B, max_legs)
            tgt = labels[:, offset::s][:, :max_legs].reshape(-1)
            pred = logits[key].reshape(-1, logits[key].size(-1))
            loss = loss + ce(pred, tgt)
        return loss / len(keys)

    elif model_type == "station":
        logits_flat = logits["station"]
        max_len = logits_flat.size(1)
        if labels.size(1) < max_len:
            pad = labels.new_full(
                (labels.size(0), max_len - labels.size(1)),
                PAD,
            )
            labels = torch.cat([labels, pad], dim=1)
        elif labels.size(1) > max_len:
            labels = labels[:, :max_len]
        return ce(
            logits_flat.reshape(-1, logits_flat.size(-1)),
            labels.reshape(-1),
        )

    raise ValueError(model_type)


def _greedy_as_beam(
    logits: dict,
    model_type: str,
    device: torch.device,
) -> list[list[tuple[torch.Tensor, float]]]:
    """Wrap greedy argmax predictions as single-beam results — vectorized."""
    if model_type in ("line", "change"):
        # (B, max_legs)
        line_preds = logits["line"].argmax(-1)
        dir_preds = logits["dir"].argmax(-1)
        B, max_legs = line_preds.shape

        if model_type == "change":
            st_preds = logits["station"].argmax(-1)  # (B, max_legs)
            # Interleave: (B, max_legs, 3) → (B, max_legs*3)
            stacked = torch.stack([line_preds, dir_preds, st_preds], dim=-1)
        else:
            stacked = torch.stack([line_preds, dir_preds], dim=-1)

        flat = stacked.reshape(B, -1)  # (B, max_legs*stride)
        return [[(flat[b], 0.0)] for b in range(B)]

    elif model_type == "station":
        preds = logits["station"].argmax(-1)  # (B, max_len)
        return [[(preds[b], 0.0)] for b in range(preds.size(0))]

    raise ValueError(model_type)


def _n_legs(labels: torch.Tensor, model_type: str) -> torch.Tensor:
    """(B,) tensor: number of legs for line/change, or non-pad token count for station."""
    if model_type == "station":
        return (labels != PAD).sum(dim=1)
    stride = 2 if model_type == "line" else 3
    return (labels[:, 0::stride] != PAD).sum(dim=1)


def _aggregate_metrics(all_metrics: list[RouteMetrics]) -> RouteMetrics:
    total_n = sum(m.n_examples for m in all_metrics)
    if total_n == 0:
        return all_metrics[0]

    def _wavg(attr: str) -> float | None:
        vals = [
            (getattr(m, attr), m.n_examples)
            for m in all_metrics
            if getattr(m, attr) is not None
        ]
        if not vals:
            return None
        return sum(v * n for v, n in vals) / sum(n for _, n in vals)

    # Merge stratified dicts across batches
    from collections import defaultdict

    merged_strat: dict[int, list[tuple[float, int]]] = defaultdict(list)
    has_strat = False
    for m in all_metrics:
        if m.stratified is not None:
            has_strat = True
            for k, (acc, n) in m.stratified.items():
                merged_strat[k].append((acc, n))
    stratified = None
    if has_strat:
        stratified = {
            k: (sum(a * n for a, n in v) / sum(n for _, n in v), sum(n for _, n in v))
            for k, v in sorted(merged_strat.items())
        }

    return RouteMetrics(
        exact_match=_wavg("exact_match"),
        any_in_beam=_wavg("any_in_beam"),
        line_acc=_wavg("line_acc"),
        dir_acc=_wavg("dir_acc"),
        station_acc=_wavg("station_acc"),
        topologically_valid=_wavg("topologically_valid"),
        n_examples=total_n,
        stratified=stratified,
    )


def _try_compile(model: nn.Module) -> nn.Module:
    try:
        compiled = torch.compile(model, mode="default")
        print("  torch.compile(mode='default') enabled")
        return compiled
    except Exception as e:
        print(f"  torch.compile unavailable ({e}), using eager mode")
        return model


def train(cfg: TrainConfig) -> None:
    import time

    from rich import print as rprint
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    t_start = time.monotonic()
    is_nexthop = cfg.model_type == "nexthop"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.manual_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rprint("[bold]Extracting topology...")
    topo = extract(cfg.gtfs_path)
    rprint(
        f"  {topo.n_stations} stations, {topo.n_lines} lines, "
        f"{len(topo.interchanges)} interchanges"
    )

    if not cfg.routes_path.exists():
        from .routes import build_dataset

        rprint("[bold]Building route dataset...")
        build_dataset(
            topo,
            max_transfers=cfg.max_transfers,
            max_results=cfg.max_routes_per_od,
            transfer_penalty=cfg.transfer_penalty,
            output_path=cfg.routes_path,
        )

    graph = build_enriched_graph(topo).to(device)

    # ── Dataset & splits ──────────────────────────────────────
    split_gen = torch.Generator(device=device).manual_seed(cfg.seed)

    if is_nexthop:
        nh_ds = NextHopGPUDataset(cfg.routes_path, device=device)
        n_stations = nh_ds.n_stations
        n_lines = nh_ds.n_lines
        stations = nh_ds.stations

        n_val_od = max(1, int(cfg.val_split * nh_ds.n_od))
        n_train_od = nh_ds.n_od - n_val_od
        od_perm = torch.randperm(nh_ds.n_od, device=device, generator=split_gen)
        train_od = od_perm[:n_train_od]
        val_od = od_perm[n_train_od:]

        train_idx = nh_ds.steps_for_ods(train_od)
        val_idx = nh_ds.steps_for_ods(val_od)
        n_train = train_idx.size(0)
        n_val = val_idx.size(0)

        rprint(
            f"  {nh_ds.n_od:,} OD pairs → {nh_ds.n_steps:,} next-hop steps"
            f"\n  {n_train_od:,} train / {n_val_od:,} val OD pairs"
            f"\n  {n_train:,} train / {n_val:,} val steps"
        )
    else:
        ds = GPURouteDataset(cfg.routes_path, model=cfg.model_type, device=device)
        n_stations = ds.n_stations
        n_lines = ds.n_lines
        stations = ds.stations

        n_total = ds.n
        n_val = max(1, int(cfg.val_split * n_total))
        n_train = n_total - n_val
        perm = torch.randperm(n_total, device=device, generator=split_gen)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        rprint(f"  {n_train:,} train, {n_val:,} val examples (GPU-resident)")

    effective_bs = min(cfg.batch_size, n_train)
    n_train_batches = (n_train + effective_bs - 1) // effective_bs
    rprint(f"  batch_size={effective_bs:,} → {n_train_batches} batches/epoch")

    # ── Model ─────────────────────────────────────────────────
    model = RouteModel(
        n_stations=n_stations,
        n_lines=n_lines,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_enc_layers=cfg.n_enc_layers,
        model_type=cfg.model_type,
        max_seq=cfg.max_seq,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    rprint(
        f"  Model [bold cyan]{cfg.model_type}[/]: {n_params:,} params | {cfg.hp_tag}"
    )

    if cfg.model_type in ("station", "nexthop"):
        adj = build_adj_mask(topo, stations).to(device)
        model.decoder.set_adj_mask(adj)
        n_avg = adj.float().sum(1).mean().item()
        rprint(
            f"  adjacency mask: {adj.sum().item():,} edges, {n_avg:.1f} avg neighbors"
        )

    if cfg.model_type == "change":
        ls_mask = build_line_station_mask(topo, stations, ds.lines).to(device)
        model.decoder.set_line_station_mask(ls_mask)
        avg_st = ls_mask.float().sum(1).mean().item()
        rprint(f"  line→station mask: {avg_st:.1f} avg stations per line")

    raw_model = model
    model = _try_compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = n_train_batches * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=max(warmup_steps, 1),
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(total_steps - warmup_steps, 1),
            ),
        ],
        milestones=[warmup_steps],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = 0.0
    best_metrics = None
    cfg.checkpoint_dir.mkdir(exist_ok=True)
    logger = MetricsLogger(cfg.model_type, cfg.hp_tag, cfg.checkpoint_dir.parent)

    train_gen = torch.Generator(device=device).manual_seed(cfg.seed)
    ce_nexthop = nn.CrossEntropyLoss() if is_nexthop else None

    # ── Progress columns adapt to model type ──────────────────
    if is_nexthop:
        progress_columns = [
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            TextColumn("·"),
            TextColumn("loss [cyan]{task.fields[train_loss]:.4f}"),
            TextColumn("step_acc [bold green]{task.fields[step_acc]:.1%}"),
            TextColumn("success [bold yellow]{task.fields[success]:.1%}"),
            TextColumn("len_ratio {task.fields[len_ratio]:.2f}"),
            TextColumn("lr [dim]{task.fields[lr]:.1e}"),
            TextColumn("{task.fields[star]}"),
        ]
        task_fields = dict(
            train_loss=0.0,
            step_acc=0.0,
            success=0.0,
            len_ratio=0.0,
            lr=cfg.lr,
            star="",
        )
    else:
        progress_columns = [
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            TextColumn("·"),
            TextColumn(
                "loss [cyan]{task.fields[train_loss]:.4f}[/]"
                "/[magenta]{task.fields[val_loss]:.4f}"
            ),
            TextColumn("top1 [bold green]{task.fields[exact_match]:.1%}"),
            TextColumn("beam {task.fields[beam_display]}"),
            TextColumn("valid {task.fields[valid]:.0%}"),
            TextColumn("lr [dim]{task.fields[lr]:.1e}"),
            TextColumn("{task.fields[star]}"),
        ]
        task_fields = dict(
            train_loss=0.0,
            val_loss=0.0,
            exact_match=0.0,
            beam_display="[dim]—",
            valid=0.0,
            lr=cfg.lr,
            star="",
        )

    with Progress(*progress_columns, refresh_per_second=4) as progress:
        epoch_task = progress.add_task("Training", total=cfg.epochs, **task_fields)

        for epoch in range(1, cfg.epochs + 1):
            shuffle = torch.randperm(n_train, device=device, generator=train_gen)
            shuffled_train = train_idx[shuffle]

            if not is_nexthop:
                ds.resample_labels()

            # ── train ─────────────────────────────────────────
            model.train()
            epoch_loss = torch.zeros(1, device=device)

            for batch_start in range(0, n_train, cfg.batch_size):
                batch_idx = shuffled_train[batch_start : batch_start + cfg.batch_size]

                if is_nexthop:
                    currents, dests_b, targets = nh_ds.get_step_batch(batch_idx)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(
                            graph.x,
                            graph.edge_index,
                            graph.edge_attr,
                            currents,
                            dests_b,
                        )
                        loss = ce_nexthop(logits["next_station"], targets)
                    n_items = currents.size(0)
                else:
                    raw_indices, origins, dests, labels = ds.get_batch(batch_idx)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(
                            graph.x,
                            graph.edge_index,
                            graph.edge_attr,
                            origins,
                            dests,
                            labels=labels,
                            sampling_p=cfg.scheduled_sampling,
                        )
                        if cfg.model_type == "station":
                            all_valid = ds.get_all_labels_batch(raw_indices)
                            loss = _compute_min_route_loss(
                                logits,
                                all_valid,
                                label_smoothing=cfg.label_smoothing,
                            )
                        else:
                            loss = _compute_loss(
                                logits,
                                labels,
                                cfg.model_type,
                                label_smoothing=cfg.label_smoothing,
                            )
                    n_items = origins.size(0)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_loss += loss.detach() * n_items

            avg_train = epoch_loss.item() / n_train

            # ── validate ──────────────────────────────────────
            model.eval()
            lr_now = optimizer.param_groups[0]["lr"]

            if is_nexthop:
                step_correct, step_total = 0, 0
                with torch.no_grad():
                    for batch_start in range(0, n_val, cfg.batch_size):
                        batch_idx = val_idx[batch_start : batch_start + cfg.batch_size]
                        currents, dests_b, targets = nh_ds.get_step_batch(batch_idx)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            logits = model(
                                graph.x,
                                graph.edge_index,
                                graph.edge_attr,
                                currents,
                                dests_b,
                            )
                        c, t = compute_nexthop_step_metrics(
                            logits["next_station"],
                            targets,
                        )
                        step_correct += c
                        step_total += t

                step_acc = step_correct / max(step_total, 1)

                # Rollout eval (periodic)
                run_rollout = (
                    cfg.beam_eval_interval > 0 and epoch % cfg.beam_eval_interval == 0
                ) or epoch == cfg.epochs

                success_rate = 0.0
                len_ratio = 0.0

                if run_rollout:
                    n_eval_od = max(1, int(n_val_od * cfg.beam_eval_sample))
                    eval_od_perm = torch.randperm(n_val_od, device=device)[:n_eval_od]
                    eval_od_idx = val_od[eval_od_perm]

                    all_rollouts: list[list[int]] = []
                    all_gt: list[list[list[int]]] = []
                    all_dests_list: list[int] = []
                    strat_keys: list[int] = []

                    rollout_bs = min(256, n_eval_od)
                    for rb_start in range(0, n_eval_od, rollout_bs):
                        rb_idx = eval_od_idx[rb_start : rb_start + rollout_bs]
                        origins_b, dests_b, gt_routes = nh_ds.get_od_batch(rb_idx)
                        rollouts = rollout_nexthop(
                            raw_model,
                            graph.x,
                            graph.edge_index,
                            graph.edge_attr,
                            origins_b,
                            dests_b,
                            max_steps=cfg.max_seq,
                        )
                        all_rollouts.extend(rollouts)
                        all_gt.extend(gt_routes)
                        all_dests_list.extend(dests_b.tolist())
                        for gt in gt_routes:
                            strat_keys.append(min(len(r) for r in gt))

                    rollout_metrics = compute_nexthop_rollout_metrics(
                        all_rollouts,
                        torch.tensor(all_dests_list, device=device),
                        all_gt,
                        strat_keys=strat_keys,
                    )
                    rollout_metrics.step_acc = step_acc
                    rollout_metrics.n_steps = step_total
                    success_rate = rollout_metrics.rollout_success
                    len_ratio = rollout_metrics.avg_length_ratio

                    if rollout_metrics.stratified:
                        bucket_ranges = [(2, 5), (6, 10), (11, 20), (21, 30), (31, 50)]
                        bucket_parts = []
                        for lo, hi in bucket_ranges:
                            total_n = 0
                            total_succ = 0
                            for k, (succ, _lr, n) in rollout_metrics.stratified.items():
                                if lo <= k <= hi:
                                    total_n += n
                                    total_succ += succ * n
                            if total_n > 0:
                                bucket_parts.append(
                                    f"{lo}-{hi}st:{total_succ / total_n:.0%}({total_n})"
                                )
                        rprint(f"  stratified success: {' | '.join(bucket_parts)}")

                star = ""
                if run_rollout and success_rate > best_val:
                    best_val = success_rate
                    best_metrics = rollout_metrics
                    star = "[bold green]★[/]"
                    ckpt = cfg.checkpoint_dir / "model_nexthop_best.pt"
                    torch.save(raw_model.state_dict(), ckpt)

                progress.update(
                    epoch_task,
                    advance=1,
                    train_loss=avg_train,
                    step_acc=step_acc,
                    success=success_rate,
                    len_ratio=len_ratio,
                    lr=lr_now,
                    star=star,
                )

            else:
                epoch_val_loss = torch.zeros(1, device=device)
                batch_metrics: list[RouteMetrics] = []

                run_beam = (
                    cfg.beam_eval_interval > 0 and epoch % cfg.beam_eval_interval == 0
                ) or epoch == cfg.epochs

                if run_beam and cfg.beam_eval_sample < 1.0:
                    n_beam_val = max(1, int(n_val * cfg.beam_eval_sample))
                    beam_perm = torch.randperm(n_val, device=device)[:n_beam_val]
                    beam_val_idx = val_idx[beam_perm]
                else:
                    beam_val_idx = val_idx

                n_beam_eval = beam_val_idx.size(0) if run_beam else n_val
                n_val_batches = (n_beam_eval + cfg.batch_size - 1) // cfg.batch_size

                if run_beam:
                    beam_progress = Progress(
                        BarColumn(),
                        "[progress.percentage]{task.percentage:>3.0f}%",
                        TimeRemainingColumn(),
                        TextColumn("·"),
                        TextColumn("beam eval"),
                        transient=True,
                    )
                    beam_progress.start()
                    beam_task = beam_progress.add_task("Beam", total=n_val_batches)
                else:
                    beam_progress = None

                with torch.no_grad():
                    for batch_start in range(0, n_beam_eval, cfg.batch_size):
                        eval_source = beam_val_idx if run_beam else val_idx
                        batch_idx = eval_source[
                            batch_start : batch_start + cfg.batch_size
                        ]
                        raw_indices, origins, dests, labels = ds.get_batch(batch_idx)

                        with torch.amp.autocast("cuda", enabled=use_amp):
                            logits = model(
                                graph.x,
                                graph.edge_index,
                                graph.edge_attr,
                                origins,
                                dests,
                                labels=labels,
                            )
                            vl = _compute_loss(
                                logits,
                                labels,
                                cfg.model_type,
                                label_smoothing=0.0,
                            )
                        epoch_val_loss += vl.detach() * origins.size(0)

                        all_valid = ds.get_all_labels_batch(raw_indices)

                        if run_beam:
                            beam_results = beam_decode(
                                raw_model,
                                graph.x,
                                graph.edge_index,
                                graph.edge_attr,
                                origins,
                                dests,
                                beam_width=cfg.beam_width,
                            )
                            beam_progress.update(beam_task, advance=1)
                        else:
                            beam_results = _greedy_as_beam(
                                logits,
                                cfg.model_type,
                                device,
                            )

                        strat_keys = _n_legs(labels, cfg.model_type)

                        batch_metrics.append(
                            compute_metrics(
                                beam_results,
                                cfg.model_type,
                                n_lines=n_lines,
                                n_stations=n_stations,
                                all_valid_labels=all_valid,
                                strat_keys=strat_keys,
                            )
                        )

                if beam_progress is not None:
                    beam_progress.stop()

                avg_val = epoch_val_loss.item() / n_val
                epoch_metrics = _aggregate_metrics(batch_metrics)

                star = ""
                if epoch_metrics.exact_match > best_val:
                    best_val = epoch_metrics.exact_match
                    best_metrics = epoch_metrics
                    star = "[bold green]★[/]"
                    ckpt = cfg.checkpoint_dir / f"model_{cfg.model_type}_best.pt"
                    torch.save(raw_model.state_dict(), ckpt)

                beam_str = (
                    f"[bold yellow]{epoch_metrics.any_in_beam:.1%}"
                    if run_beam
                    else "[dim]—"
                )

                progress.update(
                    epoch_task,
                    advance=1,
                    train_loss=avg_train,
                    val_loss=avg_val,
                    exact_match=epoch_metrics.exact_match,
                    beam_display=beam_str,
                    valid=epoch_metrics.topologically_valid,
                    lr=lr_now,
                    star=star,
                )
                logger.log(epoch, avg_train, avg_val, epoch_metrics, run_beam)

                if run_beam and epoch_metrics.stratified:
                    if cfg.model_type == "station":
                        bucket_ranges = [
                            (2, 5),
                            (6, 10),
                            (11, 20),
                            (21, 30),
                            (31, 50),
                        ]
                        bucket_parts = []
                        for lo, hi in bucket_ranges:
                            total_n = 0
                            total_correct = 0
                            for k, (acc, n) in epoch_metrics.stratified.items():
                                if lo <= k <= hi:
                                    total_n += n
                                    total_correct += acc * n
                            if total_n > 0:
                                bucket_parts.append(
                                    f"{lo}-{hi}st:{total_correct / total_n:.0%}"
                                    f"({total_n})"
                                )
                        rprint(f"  stratified: {' | '.join(bucket_parts)}")
                    else:
                        parts = [
                            f"{k}legs:{acc:.0%}({n})"
                            for k, (acc, n) in epoch_metrics.stratified.items()
                        ]
                        rprint(f"  stratified: {' | '.join(parts)}")

    elapsed = time.monotonic() - t_start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    time_str = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

    if is_nexthop:
        rprint(
            f"\n[bold green]Done.[/] Best rollout success: {best_val:.1%} ({time_str})"
        )
    else:
        rprint(f"\n[bold green]Done.[/] Best exact match: {best_val:.4f} ({time_str})")
    if best_metrics:
        rprint(f"Best metrics: {best_metrics}")
    rprint(f"Checkpoint → {cfg.checkpoint_dir / f'model_{cfg.model_type}_best.pt'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_TYPES), default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--d-model", type=int, default=None)
    args = p.parse_args()

    cfg = TrainConfig.from_defaults(
        model_type=args.model,
        profile=args.profile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
    )
    print(f"Config: {cfg.hp_tag}")
    train(cfg)
