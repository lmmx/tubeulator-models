"""Training loop for route-prediction models."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .beam import beam_decode
from .config import TrainConfig
from .dataset import PAD, RouteDataset, collate_routes
from .defaults import MODEL_TYPES, default_model_type
from .evaluate import RouteMetrics, compute_metrics
from .graph_enriched import build_enriched_graph
from .metrics_log import MetricsLogger
from .models.combined import RouteModel
from .topology import extract


__all__ = ["train"]


def _compute_loss(
    logits: dict,
    labels: torch.Tensor,
    model_type: str,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=label_smoothing)
    stride = {"line": 2, "change": 3, "station": 1}

    if model_type in ("line", "change"):
        keys = ["line", "dir"] if model_type == "line" else ["line", "dir", "station"]
        s = stride[model_type]
        max_legs = logits["line"].size(1)
        loss = torch.tensor(0.0, device=labels.device)
        count = 0
        for step in range(max_legs):
            for offset, key in enumerate(keys):
                col = step * s + offset
                if col < labels.size(1):
                    loss = loss + ce(logits[key][:, step], labels[:, col])
                    count += 1
        return loss / max(count, 1)

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
    """Wrap greedy argmax predictions as single-beam results."""
    if model_type in ("line", "change"):
        line_preds = logits["line"].argmax(-1)  # (B, max_legs)
        dir_preds = logits["dir"].argmax(-1)
        st_preds = (
            logits.get("station", logits["line"]).argmax(-1)
            if model_type == "change"
            else None
        )

        B = line_preds.size(0)
        results = []
        for b in range(B):
            flat = []
            for step in range(line_preds.size(1)):
                flat.append(line_preds[b, step].item())
                flat.append(dir_preds[b, step].item())
                if model_type == "change":
                    flat.append(st_preds[b, step].item())
            seq = torch.tensor(flat, dtype=torch.long, device=device)
            results.append([(seq, 0.0)])
        return results

    elif model_type == "station":
        preds = logits["station"].argmax(-1)  # (B, max_len)
        return [[(preds[b], 0.0)] for b in range(preds.size(0))]

    raise ValueError(model_type)


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

    return RouteMetrics(
        exact_match=_wavg("exact_match"),
        any_in_beam=_wavg("any_in_beam"),
        line_acc=_wavg("line_acc"),
        dir_acc=_wavg("dir_acc"),
        station_acc=_wavg("station_acc"),
        topologically_valid=_wavg("topologically_valid"),
        n_examples=total_n,
    )


def train(cfg: TrainConfig) -> None:
    from rich import print as rprint
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

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

    ds = RouteDataset(cfg.routes_path, model=cfg.model_type)
    n_val = max(1, int(cfg.val_split * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_routes,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        collate_fn=collate_routes,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    rprint(f"  {n_train:,} train, {n_val:,} val examples")

    model = RouteModel(
        n_stations=ds.n_stations,
        n_lines=ds.n_lines,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_dl) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=warmup_steps,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
            ),
        ],
        milestones=[warmup_steps],
    )

    best_val = float("inf")
    best_metrics: RouteMetrics | None = None
    cfg.checkpoint_dir.mkdir(exist_ok=True)
    logger = MetricsLogger(cfg.model_type, cfg.hp_tag, cfg.checkpoint_dir.parent)

    with Progress(
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
        refresh_per_second=4,
    ) as progress:
        epoch_task = progress.add_task(
            "Training",
            total=cfg.epochs,
            train_loss=0.0,
            val_loss=0.0,
            exact_match=0.0,
            beam_display="[dim]—",
            valid=0.0,
            lr=cfg.lr,
            star="",
        )

        for epoch in range(1, cfg.epochs + 1):
            # ── train (unchanged) ─────────────────────────────
            model.train()
            total_loss = 0.0
            for _indices, origins, dests, labels in train_dl:
                origins = origins.to(device)
                dests = dests.to(device)
                labels = labels.to(device)

                logits = model(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    origins,
                    dests,
                    labels=labels,
                    sampling_p=cfg.scheduled_sampling,
                )
                loss = _compute_loss(
                    logits,
                    labels,
                    cfg.model_type,
                    label_smoothing=cfg.label_smoothing,
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * origins.size(0)

            avg_train = total_loss / n_train

            # ── validate ──────────────────────────────────────
            model.eval()
            val_loss = 0.0
            batch_metrics: list[RouteMetrics] = []

            run_beam = (
                cfg.beam_eval_interval > 0 and epoch % cfg.beam_eval_interval == 0
            ) or epoch == cfg.epochs  # always on final epoch

            with torch.no_grad():
                for indices, origins, dests, labels in val_dl:
                    origins = origins.to(device)
                    dests = dests.to(device)
                    labels = labels.to(device)

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
                    val_loss += vl.item() * origins.size(0)

                    all_valid = [
                        [lbl.to(device) for lbl in ds.get_all_labels(i)]
                        for i in indices
                    ]

                    if run_beam:
                        beam_results = beam_decode(
                            model,
                            graph.x,
                            graph.edge_index,
                            graph.edge_attr,
                            origins,
                            dests,
                            beam_width=cfg.beam_width,
                        )
                    else:
                        # Greedy: top-1 from argmax, wrapped as fake beam
                        beam_results = _greedy_as_beam(
                            logits,
                            cfg.model_type,
                            device,
                        )

                    batch_metrics.append(
                        compute_metrics(
                            beam_results,
                            cfg.model_type,
                            n_lines=ds.n_lines,
                            n_stations=ds.n_stations,
                            all_valid_labels=all_valid,
                        )
                    )
            avg_val = val_loss / n_val
            epoch_metrics = _aggregate_metrics(batch_metrics)

            lr_now = optimizer.param_groups[0]["lr"]

            star = ""
            if avg_val < best_val:
                best_val = avg_val
                best_metrics = epoch_metrics
                star = "[bold green]★[/]"
                ckpt = cfg.checkpoint_dir / f"model_{cfg.model_type}_best.pt"
                torch.save(model.state_dict(), ckpt)

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

    rprint(f"\n[bold green]Done.[/] Best val loss: {best_val:.4f}")
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
