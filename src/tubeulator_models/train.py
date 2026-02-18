"""Training loop for route-prediction models."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .config import TrainConfig
from .dataset import PAD, RouteDataset, collate_routes
from .defaults import MODEL_TYPES
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    print("Extracting topology...")
    topo = extract(cfg.gtfs_path)
    print(
        f"  {topo.n_stations} stations, {topo.n_lines} lines, "
        f"{len(topo.interchanges)} interchanges"
    )

    if not cfg.routes_path.exists():
        from .routes import build_dataset

        print("Building route dataset...")
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
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        collate_fn=collate_routes,
    )
    print(f"  {n_train:,} train, {n_val:,} val examples")

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
    print(f"  Model {cfg.model_type}: {n_params:,} params | {cfg.hp_tag}")

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

        model.eval()
        val_loss = 0.0
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
        avg_val = val_loss / n_val

        if epoch % cfg.log_every == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  epoch {epoch:3d} | train {avg_train:.4f} "
                f"| val {avg_val:.4f} | lr {lr_now:.2e}"
            )

        if avg_val < best_val:
            best_val = avg_val
            ckpt = cfg.checkpoint_dir / f"model_{cfg.model_type}_best.pt"
            torch.save(model.state_dict(), ckpt)

    print(f"Done. Best val loss: {best_val:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_TYPES), default="change")
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
