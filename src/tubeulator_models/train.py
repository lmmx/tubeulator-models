"""Training loop for route-prediction models."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .config import TrainConfig
from .dataset import PAD, RouteDataset, collate_routes
from .defaults import MODEL_TYPES
from .evaluate import RouteMetrics, compute_metrics
from .graph_enriched import build_enriched_graph
from .models.combined import RouteModel
from .topology import extract


__all__ = ["train"]


def _compute_loss(
    logits: dict,
    labels: torch.Tensor,
    model_type: str,
) -> torch.Tensor:
    ce = nn.CrossEntropyLoss(ignore_index=PAD)
    stride = {"line": 2, "change": 3}

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


def train(cfg: TrainConfig) -> None:
    from rich import print as rprint
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
    )

    best_val = float("inf")
    cfg.checkpoint_dir.mkdir(exist_ok=True)

    with Progress(
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        TextColumn("·"),
        TextColumn(
            "loss [cyan]{task.fields[train_loss]:.4f}[/]/[magenta]{task.fields[val_loss]:.4f}"
        ),
        TextColumn("exact [bold green]{task.fields[exact_match]:.1%}"),
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
            lr=cfg.lr,
            star="",
        )

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            total_loss = 0.0
            for origins, dests, labels in train_dl:
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
                loss = _compute_loss(logits, labels, cfg.model_type)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                total_loss += loss.item() * origins.size(0)

            scheduler.step()
            avg_train = total_loss / n_train

            # --- validation ---
            model.eval()
            val_loss = 0.0
            all_metrics: list[RouteMetrics] = []
            with torch.no_grad():
                for origins, dests, labels in val_dl:
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
                    vl = _compute_loss(logits, labels, cfg.model_type)
                    val_loss += vl.item() * origins.size(0)

                    all_metrics.append(
                        compute_metrics(
                            logits,
                            labels,
                            cfg.model_type,
                            n_lines=ds.n_lines,
                            n_stations=ds.n_stations,
                        )
                    )
            avg_val = val_loss / n_val

            # aggregate metrics across batches
            total_em = sum(m.exact_match * m.n_examples for m in all_metrics)
            total_n = sum(m.n_examples for m in all_metrics)
            epoch_em = total_em / max(total_n, 1)

            star = ""
            if avg_val < best_val:
                best_val = avg_val
                star = "[bold green]★[/]"
                ckpt = cfg.checkpoint_dir / f"model_{cfg.model_type}_best.pt"
                torch.save(model.state_dict(), ckpt)

            progress.update(
                epoch_task,
                advance=1,
                train_loss=avg_train,
                val_loss=avg_val,
                exact_match=epoch_em,
                lr=scheduler.get_last_lr()[0],
                star=star,
            )

    rprint(f"\n[bold green]Done.[/] Best val loss: {best_val:.4f}")
    if all_metrics:
        final = all_metrics[-1]  # last epoch's last batch (approximate)
        rprint(f"Final metrics: {final}")
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
