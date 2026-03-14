"""Training loop for route-prediction models."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .beam import beam_rollout_nexthop, bellman_rollout_nexthop
from .config import TrainConfig
from .dataset import PAD, NextHopGPUDataset
from .defaults import MODEL_TYPES, repo_root, resolve_data
from .evaluate import compute_nexthop_rollout_metrics, compute_nexthop_step_metrics
from .graph_enriched import build_enriched_graph
from .models.combined import RouteModel
from .topology import (
    build_adj_mask,
    build_edge_time_matrix,
    extract,
    floyd_warshall_line_aware,
    floyd_warshall_times,
    load_interchange_data,
)


__all__ = ["train"]


# The 0.1 weight on value loss is a starting point: value pred
# is auxiliary and shouldn't dominate the policy gradient.
# MSE loss scale is naturally larger (predicting numbers like
# 5-40 vs cross-entropy around 0.2), so 0.1 keeps them balanced
VALUE_LOSS_WEIGHT = 0.5


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


def _n_legs(labels: torch.Tensor, model_type: str) -> torch.Tensor:
    """(B,) tensor: number of legs for line/change, or non-pad token count for station."""
    if model_type == "station":
        return (labels != PAD).sum(dim=1)
    stride = 2 if model_type == "line" else 3
    return (labels[:, 0::stride] != PAD).sum(dim=1)


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
        raise ValueError("Not implemented for non-nexthop")

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
        value_primary=cfg.value_primary,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    rprint(
        f"  Model [bold cyan]{cfg.model_type}[/]: {n_params:,} params | {cfg.hp_tag}"
    )

    if is_nexthop:
        adj = build_adj_mask(topo, stations).to(device)
        model.decoder.set_adj_mask(adj)
        n_avg = adj.float().sum(1).mean().item()
        rprint(
            f"  adjacency mask: {adj.sum().item():,} edges, {n_avg:.1f} avg neighbors"
        )

    edge_time_matrix = None
    optimal_times = None
    optimal_times_eval = None
    q_matrix = None
    if is_nexthop:
        edge_time_matrix = build_edge_time_matrix(topo, stations).to(device)
        lines = nh_ds.lines

        data_cfg = resolve_data()
        ic_rel = data_cfg.get("interchange_path", "")
        ic_path = repo_root() / ic_rel if ic_rel else None

        if ic_path is not None and ic_path.is_file():
            interchange_data = load_interchange_data(ic_path)
            optimal_times, q_matrix, optimal_times_eval = floyd_warshall_line_aware(
                topo,
                stations,
                lines,
                interchange_data,
                discount=cfg.transfer_discount,
            )
            optimal_times = optimal_times.to(device)
            q_matrix = q_matrix.to(device)
            optimal_times_eval = optimal_times_eval.to(device)
            rprint(f"  line-aware Floyd-Warshall (discount={cfg.transfer_discount})")
        else:
            optimal_times_eval = floyd_warshall_times(topo, stations).to(device)
            optimal_times = optimal_times_eval
            rprint("  transfer-free Floyd-Warshall (no interchange data)")

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

    # ── Value-primary: separate training path ─────────────────
    if is_nexthop and cfg.value_primary:
        _train_value_primary(
            cfg,
            model,
            raw_model,
            graph,
            device,
            use_amp,
            n_stations,
            stations,
            topo,
            edge_time_matrix,
            optimal_times,
            optimal_times_eval,
            split_gen,
        )
        return

    # ── Eval-only: load checkpoint and run full rollout ───────
    if cfg.eval_only:
        ckpt = cfg.checkpoint_dir / f"model_{cfg.model_type}_best.pt"
        if not ckpt.exists():
            rprint(f"[red]No checkpoint found at {ckpt}[/]")
            return
        raw_model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True)
        )
        rprint(f"  Loaded checkpoint: {ckpt}")

        if is_nexthop:
            model.eval()
            all_rollouts = []
            all_gt = []
            all_origins_list = []
            all_dests_list = []
            strat_keys = []

            rollout_bs = min(256, n_val_od)
            for rb_start in range(0, n_val_od, rollout_bs):
                rb_idx = val_od[rb_start : rb_start + rollout_bs]
                origins_b, dests_b, gt_routes = nh_ds.get_od_batch(rb_idx)
                rollouts = beam_rollout_nexthop(
                    raw_model,
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    origins_b,
                    dests_b,
                    beam_width=cfg.beam_width,
                    max_steps=cfg.max_seq,
                )
                all_rollouts.extend([beams[0][0] for beams in rollouts])
                all_gt.extend(gt_routes)
                all_origins_list.extend(origins_b.tolist())
                all_dests_list.extend(dests_b.tolist())
                for gt in gt_routes:
                    strat_keys.append(min(len(r) for r in gt))

            # Find the inf culprit
            for b, route in enumerate(all_rollouts):
                if route[-1] != all_dests_list[b]:
                    continue
                for i in range(len(route) - 1):
                    t = edge_time_matrix[route[i], route[i + 1]].item()
                    if t == float("inf"):
                        rprint(
                            f"  [red]INF EDGE: route {b}, "
                            f"edge {route[i]}→{route[i + 1]}, "
                            f"stations={stations[route[i]]}→{stations[route[i + 1]]}[/]"
                        )
                        break

            metrics = compute_nexthop_rollout_metrics(
                all_rollouts,
                torch.tensor(all_dests_list, device=device),
                all_gt,
                edge_time_matrix=edge_time_matrix,
                optimal_times=optimal_times_eval,
                origins=torch.tensor(all_origins_list, device=device),
                strat_keys=strat_keys,
            )
            rprint(f"\n[bold]Eval on full val set ({n_val_od:,} OD pairs):[/]")
            rprint(f"  {metrics}")

            if metrics.stratified:
                bucket_ranges = [(2, 5), (6, 10), (11, 20), (21, 30), (31, 50)]
                bucket_parts = []
                for lo, hi in bucket_ranges:
                    total_n = 0
                    total_succ = 0
                    dij_vals = []
                    for k, (succ, _lr, dij, n) in metrics.stratified.items():
                        if lo <= k <= hi:
                            total_n += n
                            total_succ += succ * n
                            if dij != float("inf"):
                                dij_vals.extend([dij] * n)
                    if total_n > 0:
                        avg_dij = (
                            sum(dij_vals) / len(dij_vals) if dij_vals else float("inf")
                        )
                        bucket_parts.append(
                            f"{lo}-{hi}st:{total_succ / total_n:.0%}"
                            f" dij={avg_dij:.2f}({total_n})"
                        )
                rprint(f"  stratified: {' | '.join(bucket_parts)}")

            # ── Bellman rollout diagnostic ────────────────────────
            rprint("\n[bold]Bellman rollout diagnostic:[/]")

            all_bellman_routes = []
            all_gt_bell = []
            all_origins_bell = []
            all_dests_bell = []
            strat_keys_bell = []

            for rb_start in range(0, n_val_od, rollout_bs):
                rb_idx = val_od[rb_start : rb_start + rollout_bs]
                origins_b, dests_b, gt_routes = nh_ds.get_od_batch(rb_idx)
                routes = bellman_rollout_nexthop(
                    raw_model,
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    origins_b,
                    dests_b,
                    edge_time_matrix,
                    max_steps=cfg.max_seq,
                )
                all_bellman_routes.extend(routes)
                all_gt_bell.extend(gt_routes)
                all_origins_bell.extend(origins_b.tolist())
                all_dests_bell.extend(dests_b.tolist())
                for gt in gt_routes:
                    strat_keys_bell.append(min(len(r) for r in gt))

            bellman_metrics = compute_nexthop_rollout_metrics(
                all_bellman_routes,
                torch.tensor(all_dests_bell, device=device),
                all_gt_bell,
                edge_time_matrix=edge_time_matrix,
                optimal_times=optimal_times_eval,
                origins=torch.tensor(all_origins_bell, device=device),
                strat_keys=strat_keys_bell,
            )
            rprint(f"  Bellman: {bellman_metrics}")

            if bellman_metrics.stratified:
                bucket_ranges = [(2, 5), (6, 10), (11, 20), (21, 30), (31, 50)]
                bucket_parts = []
                for lo, hi in bucket_ranges:
                    total_n = 0
                    total_succ = 0
                    dij_vals = []
                    for k, (succ, _lr, dij, n) in bellman_metrics.stratified.items():
                        if lo <= k <= hi:
                            total_n += n
                            total_succ += succ * n
                            if dij != float("inf"):
                                dij_vals.extend([dij] * n)
                    if total_n > 0:
                        avg_dij = (
                            sum(dij_vals) / len(dij_vals) if dij_vals else float("inf")
                        )
                        bucket_parts.append(
                            f"{lo}-{hi}st:{total_succ / total_n:.0%}"
                            f" dij={avg_dij:.2f}({total_n})"
                        )
                rprint(f"  stratified: {' | '.join(bucket_parts)}")

            # ── Value head MAE against Floyd-Warshall ─────────────
            rprint("\n[bold]Value head accuracy:[/]")
            model.eval()
            with torch.no_grad():
                H = raw_model.encoder(graph.x, graph.edge_index, graph.edge_attr)
                sample_od = val_od[: min(2000, n_val_od)]
                o_samp = nh_ds.od_origins[sample_od]
                d_samp = nh_ds.od_dests[sample_od]
                h_o = H[o_samp]
                h_d = H[d_samp]
                combined = torch.cat([h_o, h_d], dim=-1)
                v_pred = raw_model.decoder.value_head(combined).squeeze(-1)
                # Floyd-Warshall ground truth: remaining time from origin to dest in minutes
                v_true = (
                    optimal_times_eval[o_samp.cpu(), d_samp.cpu()].to(device) / 60.0
                )
                mae = (v_pred - v_true).abs().mean().item()
                rmse = ((v_pred - v_true) ** 2).mean().sqrt().item()
                rprint(f"  MAE:  {mae:.2f} min")
                rprint(f"  RMSE: {rmse:.2f} min")
        return

    train_gen = torch.Generator(device=device).manual_seed(cfg.seed)

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
        raise ValueError("Not implemented for non-nexthop")

    with Progress(*progress_columns, refresh_per_second=4) as progress:
        epoch_task = progress.add_task("Training", total=cfg.epochs, **task_fields)

        for epoch in range(1, cfg.epochs + 1):
            shuffle = torch.randperm(n_train, device=device, generator=train_gen)
            shuffled_train = train_idx[shuffle]

            # ── train ─────────────────────────────────────────
            model.train()
            epoch_loss = torch.zeros(1, device=device)

            for batch_start in range(0, n_train, cfg.batch_size):
                batch_idx = shuffled_train[batch_start : batch_start + cfg.batch_size]

                if is_nexthop:
                    currents, dests_b, targets, remaining = nh_ds.get_step_batch(
                        batch_idx
                    )
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(
                            graph.x,
                            graph.edge_index,
                            graph.edge_attr,
                            currents,
                            dests_b,
                        )
                        # Q-soft targets: cost-aware supervision
                        raw_logits = logits["next_station"]  # (B, N)
                        adj = model.decoder.adj_mask[currents]  # (B, N)

                        # Line-aware Q: accounts for transfer at arrival station
                        if q_matrix is not None:
                            q = q_matrix[currents]  # (B, N, N)
                            q = q.gather(
                                2, dests_b.view(-1, 1, 1).expand(-1, n_stations, 1)
                            ).squeeze(-1)  # (B, N)
                        else:
                            q = edge_time_matrix[currents] + optimal_times[:, dests_b].T
                        q = q.masked_fill(~adj, float("inf"))

                        # Self-loops and missing expanded-graph entries produce inf
                        # at adj-True positions. Replace with worst-neighbour + 10min
                        # so they're strongly disfavoured but don't poison normalization.
                        adj_inf = adj & q.isinf()
                        if adj_inf.any():
                            q_finite = q.masked_fill(q.isinf(), 0.0)
                            q_max = q_finite.max(dim=-1, keepdim=True).values  # (B, 1)
                            q = torch.where(adj_inf, q_max + 600.0, q)

                        # Center and normalize per sample
                        q_min = q.min(dim=-1, keepdim=True).values
                        q_centered = q - q_min
                        q_for_std = q_centered.clone()
                        q_for_std[~adj] = 0.0
                        n_adj = adj.float().sum(dim=-1, keepdim=True)
                        q_mean = q_for_std.sum(dim=-1, keepdim=True) / n_adj
                        q_var = ((q_for_std - q_mean) * adj.float()).pow(2).sum(
                            dim=-1, keepdim=True
                        ) / n_adj
                        q_std = q_var.sqrt().clamp(min=1.0)
                        q_norm = q_centered / q_std

                        # Soft targets via softmin
                        beta = 4.0
                        soft_targets = F.softmax(-beta * q_norm, dim=-1)

                        # KL divergence
                        log_probs = F.log_softmax(raw_logits, dim=-1)
                        policy_loss = F.kl_div(
                            log_probs, soft_targets, reduction="batchmean"
                        )
                        value_loss = F.mse_loss(logits["value"], remaining)
                        loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss
                    n_items = currents.size(0)
                else:
                    raise ValueError(f"Model type {cfg.model_type} is not nexthop")

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
                        currents, dests_b, targets, _remaining = nh_ds.get_step_batch(
                            batch_idx
                        )
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
                    all_origins_list: list[int] = []
                    all_dests_list: list[int] = []
                    strat_keys: list[int] = []

                    rollout_bs = min(256, n_eval_od)
                    for rb_start in range(0, n_eval_od, rollout_bs):
                        rb_idx = eval_od_idx[rb_start : rb_start + rollout_bs]
                        origins_b, dests_b, gt_routes = nh_ds.get_od_batch(rb_idx)
                        rollouts = beam_rollout_nexthop(
                            raw_model,
                            graph.x,
                            graph.edge_index,
                            graph.edge_attr,
                            origins_b,
                            dests_b,
                            beam_width=cfg.beam_width,
                            max_steps=cfg.max_seq,
                        )
                        all_rollouts.extend([beams[0][0] for beams in rollouts])
                        all_gt.extend(gt_routes)
                        all_origins_list.extend(origins_b.tolist())
                        all_dests_list.extend(dests_b.tolist())
                        for gt in gt_routes:
                            strat_keys.append(min(len(r) for r in gt))

                    rollout_metrics = compute_nexthop_rollout_metrics(
                        all_rollouts,
                        torch.tensor(all_dests_list, device=device),
                        all_gt,
                        edge_time_matrix=edge_time_matrix,
                        optimal_times=optimal_times_eval,
                        origins=torch.tensor(all_origins_list, device=device),
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
                            dij_vals = []
                            for k, (
                                succ,
                                _lr,
                                dij,
                                n,
                            ) in rollout_metrics.stratified.items():
                                if lo <= k <= hi:
                                    total_n += n
                                    total_succ += succ * n
                                    if dij != float("inf"):
                                        dij_vals.extend([dij] * n)
                            if total_n > 0:
                                avg_dij = (
                                    sum(dij_vals) / len(dij_vals)
                                    if dij_vals
                                    else float("inf")
                                )
                                bucket_parts.append(
                                    f"{lo}-{hi}st:{total_succ / total_n:.0%}"
                                    f" dij={avg_dij:.2f}({total_n})"
                                )
                        rprint(f"  stratified: {' | '.join(bucket_parts)}")

                star = ""
                if run_rollout and success_rate > best_val:
                    best_val = success_rate
                    best_metrics = rollout_metrics
                    star = "[bold green]★[/]"
                    ckpt = cfg.checkpoint_dir / "model_nexthop_best.pt"
                    torch.save(raw_model.state_dict(), ckpt)
                    ckpt.with_suffix(".metrics.json").write_text(
                        json.dumps(
                            {
                                "success_rate": success_rate,
                                "dijkstra_ratio": rollout_metrics.avg_dijkstra_ratio,
                                "step_acc": step_acc,
                                "len_ratio": rollout_metrics.avg_length_ratio,
                                "n_od_pairs": nh_ds.n_od,
                                "n_steps": nh_ds.n_steps,
                                "n_train_od": n_train_od,
                                "n_val_od": n_val_od,
                                "n_train_steps": n_train,
                                "n_val_steps": n_val,
                                "batch_size": cfg.batch_size,
                                "train_time_s": time.monotonic() - t_start,
                            },
                            indent=2,
                        )
                        + "\n"
                    )

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
                raise ValueError("Not implemented for non-nexthop")

    elapsed = time.monotonic() - t_start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    time_str = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

    if is_nexthop:
        rprint(
            f"\n[bold green]Done.[/] Best rollout success: {best_val:.1%} ({time_str})"
        )
    if best_metrics:
        rprint(f"Best metrics: {best_metrics}")
    rprint(f"Checkpoint → {cfg.checkpoint_dir / f'model_{cfg.model_type}_best.pt'}")


def _train_value_primary(
    cfg: TrainConfig,
    model,
    raw_model,
    graph,
    device: torch.device,
    use_amp: bool,
    n_stations: int,
    stations: list[str],
    topo,
    edge_time_matrix: torch.Tensor,
    optimal_times: torch.Tensor,
    optimal_times_eval: torch.Tensor,
    split_gen: torch.Generator,
) -> None:
    """
    Train the encoder + value head with pure MSE against Floyd-Warshall.

    No policy loss. The model learns V(s, d) = shortest travel time from s to d.
    At inference, Bellman rollout: argmin_n [ edge_time(s,n) + V(n,d) ].
    """
    import time

    from rich import print as rprint
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    t_start = time.monotonic()
    N = n_stations

    # ── Build all valid OD pairs from Floyd-Warshall ──────────
    valid_mask = optimal_times < float("inf")
    valid_mask.fill_diagonal_(False)
    all_s, all_d = valid_mask.nonzero(as_tuple=True)
    n_pairs = all_s.size(0)

    # Targets in minutes (matches existing value head convention)
    all_targets = optimal_times[all_s, all_d] / 60.0

    rprint(f"  Value-primary mode: {n_pairs:,} valid OD pairs from {N} stations")

    # ── Train/val split at OD-pair level ──────────────────────
    n_val_pairs = max(1, int(cfg.val_split * n_pairs))
    n_train_pairs = n_pairs - n_val_pairs
    pair_perm = torch.randperm(n_pairs, device=device, generator=split_gen)

    train_idx = pair_perm[:n_train_pairs]
    val_idx = pair_perm[n_train_pairs:]

    train_s, train_d = all_s[train_idx], all_d[train_idx]
    train_targets = all_targets[train_idx]
    val_s, val_d = all_s[val_idx], all_d[val_idx]
    val_targets = all_targets[val_idx]

    rprint(f"  {n_train_pairs:,} train / {n_val_pairs:,} val OD pairs")

    effective_bs = min(cfg.batch_size, n_train_pairs)
    n_batches = (n_train_pairs + effective_bs - 1) // effective_bs
    rprint(f"  batch_size={effective_bs:,} → {n_batches} batches/epoch")

    # ── Only train encoder + value head, freeze policy MLP ────
    for p in raw_model.decoder.mlp.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = n_batches * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=max(warmup_steps, 1)
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_steps - warmup_steps, 1)
            ),
        ],
        milestones=[warmup_steps],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_gen = torch.Generator(device=device).manual_seed(cfg.seed)

    # ── Load rollout eval dataset (for Bellman eval) ──────────
    nh_ds = NextHopGPUDataset(cfg.routes_path, device=device)
    n_od = nh_ds.n_od
    n_val_od = max(1, int(cfg.val_split * n_od))
    od_perm = torch.randperm(n_od, device=device, generator=split_gen)
    val_od = od_perm[n_od - n_val_od :]

    best_mae = float("inf")
    best_bellman_dij = float("inf")
    cfg.checkpoint_dir.mkdir(exist_ok=True)

    # ── Eval-only path ────────────────────────────────────────
    if cfg.eval_only:
        ckpt = cfg.checkpoint_dir / "model_nexthop_value_best.pt"
        if not ckpt.exists():
            rprint(f"[red]No checkpoint at {ckpt}[/]")
            return
        raw_model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True)
        )
        rprint(f"  Loaded: {ckpt}")
        _eval_value_primary(
            raw_model,
            model,
            graph,
            device,
            use_amp,
            val_s,
            val_d,
            val_targets,
            nh_ds,
            val_od,
            edge_time_matrix,
            optimal_times,
            optimal_times_eval,
            n_stations,
            stations,
            cfg,
            full=True,
        )
        return

    # ── Training loop ─────────────────────────────────────────
    with Progress(
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        TextColumn("·"),
        TextColumn("loss [cyan]{task.fields[loss]:.4f}"),
        TextColumn("MAE [bold green]{task.fields[mae]:.2f}[/]min"),
        TextColumn("bellman [bold yellow]{task.fields[bellman]:.2f}"),
        TextColumn("success [bold]{task.fields[success]:.1%}"),
        TextColumn("lr [dim]{task.fields[lr]:.1e}"),
        TextColumn("{task.fields[star]}"),
        refresh_per_second=4,
    ) as progress:
        task = progress.add_task(
            "Value-primary",
            total=cfg.epochs,
            loss=0.0,
            mae=99.0,
            bellman=99.0,
            success=0.0,
            lr=cfg.lr,
            star="",
        )

        for epoch in range(1, cfg.epochs + 1):
            # ── Train ─────────────────────────────────────────
            model.train()
            shuffle = torch.randperm(n_train_pairs, device=device, generator=train_gen)
            epoch_loss = 0.0
            n_seen = 0

            for batch_start in range(0, n_train_pairs, effective_bs):
                batch_perm = shuffle[batch_start : batch_start + effective_bs]
                s_batch = train_s[batch_perm]
                d_batch = train_d[batch_perm]
                t_batch = train_targets[batch_perm]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    H = model.encoder(graph.x, graph.edge_index, graph.edge_attr)
                    h_s = H[s_batch]
                    h_d = H[d_batch]
                    combined = torch.cat([h_s, h_d], dim=-1)
                    v_pred = model.decoder.value_head(combined).squeeze(-1)
                    # Errors under 2 min get squared, linear penalty for errors over that
                    loss = F.huber_loss(v_pred, t_batch, delta=2.0)  # 2 min

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                bs = s_batch.size(0)
                epoch_loss += loss.detach().item() * bs
                n_seen += bs

            avg_loss = epoch_loss / n_seen

            # ── Validate: MAE on held-out OD pairs ────────────
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                H = raw_model.encoder(graph.x, graph.edge_index, graph.edge_attr)
                h_vs = H[val_s]
                h_vd = H[val_d]
                combined = torch.cat([h_vs, h_vd], dim=-1)
                v_pred = raw_model.decoder.value_head(combined).squeeze(-1)
                mae = (v_pred - val_targets).abs().mean().item()

            # ── Bellman rollout eval (periodic) ───────────────
            run_bellman = (
                cfg.beam_eval_interval > 0 and epoch % cfg.beam_eval_interval == 0
            ) or epoch == cfg.epochs

            success = 0.0
            dij_ratio = float("inf")

            if run_bellman:
                n_eval_od = max(1, int(n_val_od * cfg.beam_eval_sample))
                eval_od_perm = torch.randperm(n_val_od, device=device)[:n_eval_od]
                eval_od_idx = val_od[eval_od_perm]

                all_routes_b: list[list[int]] = []
                all_gt: list[list[list[int]]] = []
                all_origins_l: list[int] = []
                all_dests_l: list[int] = []
                strat_keys: list[int] = []

                rollout_bs = min(256, n_eval_od)
                for rb_start in range(0, n_eval_od, rollout_bs):
                    rb_idx = eval_od_idx[rb_start : rb_start + rollout_bs]
                    origins_b, dests_b, gt_routes = nh_ds.get_od_batch(rb_idx)
                    routes = bellman_rollout_nexthop(
                        raw_model,
                        graph.x,
                        graph.edge_index,
                        graph.edge_attr,
                        origins_b,
                        dests_b,
                        edge_time_matrix,
                        max_steps=cfg.max_seq,
                    )
                    all_routes_b.extend(routes)
                    all_gt.extend(gt_routes)
                    all_origins_l.extend(origins_b.tolist())
                    all_dests_l.extend(dests_b.tolist())
                    for gt in gt_routes:
                        strat_keys.append(min(len(r) for r in gt))

                rollout_metrics = compute_nexthop_rollout_metrics(
                    all_routes_b,
                    torch.tensor(all_dests_l, device=device),
                    all_gt,
                    edge_time_matrix=edge_time_matrix,
                    optimal_times=optimal_times_eval,
                    origins=torch.tensor(all_origins_l, device=device),
                    strat_keys=strat_keys,
                )
                success = rollout_metrics.rollout_success
                dij_ratio = rollout_metrics.avg_dijkstra_ratio

                if rollout_metrics.stratified:
                    bucket_ranges = [(2, 5), (6, 10), (11, 20), (21, 30), (31, 50)]
                    bucket_parts = []
                    for lo, hi in bucket_ranges:
                        total_n = 0
                        total_succ = 0
                        dij_vals = []
                        for k, (
                            succ,
                            _lr,
                            dij,
                            n,
                        ) in rollout_metrics.stratified.items():
                            if lo <= k <= hi:
                                total_n += n
                                total_succ += succ * n
                                if dij != float("inf"):
                                    dij_vals.extend([dij] * n)
                        if total_n > 0:
                            avg_d = (
                                sum(dij_vals) / len(dij_vals)
                                if dij_vals
                                else float("inf")
                            )
                            bucket_parts.append(
                                f"{lo}-{hi}st:{total_succ / total_n:.0%}"
                                f" dij={avg_d:.2f}({total_n})"
                            )
                    rprint(f"  stratified: {' | '.join(bucket_parts)}")

            # ── Checkpoint on best MAE ────────────────────────
            lr_now = optimizer.param_groups[0]["lr"]
            star = ""
            if mae < best_mae:
                best_mae = mae
                star = "[bold green]★[/]"
                ckpt = cfg.checkpoint_dir / "model_nexthop_value_best.pt"
                torch.save(raw_model.state_dict(), ckpt)
                ckpt.with_suffix(".metrics.json").write_text(
                    json.dumps({"mae": mae}, indent=2) + "\n"
                )
            if run_bellman and dij_ratio < best_bellman_dij:
                best_bellman_dij = dij_ratio

            progress.update(
                task,
                advance=1,
                loss=avg_loss,
                mae=mae,
                bellman=dij_ratio if run_bellman else best_bellman_dij,
                success=success if run_bellman else 0.0,
                lr=lr_now,
                star=star,
            )

    elapsed = time.monotonic() - t_start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    time_str = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
    rprint(
        f"\n[bold green]Done.[/] Best MAE: {best_mae:.2f} min | "
        f"Best Bellman Dijkstra: {best_bellman_dij:.2f} ({time_str})"
    )
    rprint(f"Checkpoint → {cfg.checkpoint_dir / 'model_nexthop_value_best.pt'}")


def _eval_value_primary(
    raw_model,
    model,
    graph,
    device,
    use_amp,
    val_s,
    val_d,
    val_targets,
    nh_ds,
    val_od,
    edge_time_matrix,
    optimal_times,
    optimal_times_eval,
    n_stations,
    stations,
    cfg,
    full: bool = False,
):
    """Full eval for value-primary checkpoint."""
    from rich import print as rprint

    model.eval()
    n_val_od = val_od.size(0)

    # ── Value head MAE ────────────────────────────────────────
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        H = raw_model.encoder(graph.x, graph.edge_index, graph.edge_attr)
        h_vs = H[val_s]
        h_vd = H[val_d]
        combined = torch.cat([h_vs, h_vd], dim=-1)
        v_pred = raw_model.decoder.value_head(combined).squeeze(-1)
        mae = (v_pred - val_targets).abs().mean().item()
        rmse = ((v_pred - val_targets) ** 2).mean().sqrt().item()

    rprint("\n[bold]Value head accuracy:[/]")
    rprint(f"  MAE:  {mae:.2f} min")
    rprint(f"  RMSE: {rmse:.2f} min")

    # ── Distance distribution diagnostics ─────────────────────
    errors = (v_pred - val_targets).abs()
    for threshold in [0.5, 1.0, 2.0, 5.0]:
        pct = (errors < threshold).float().mean().item()
        rprint(f"  < {threshold} min error: {pct:.1%}")

    # ── Bellman rollout ───────────────────────────────────────
    rprint(f"\n[bold]Bellman rollout ({n_val_od:,} OD pairs):[/]")

    all_routes: list[list[int]] = []
    all_gt: list[list[list[int]]] = []
    all_origins_l: list[int] = []
    all_dests_l: list[int] = []
    strat_keys: list[int] = []

    rollout_bs = min(256, n_val_od)
    for rb_start in range(0, n_val_od, rollout_bs):
        rb_idx = val_od[rb_start : rb_start + rollout_bs]
        origins_b, dests_b, gt_routes = nh_ds.get_od_batch(rb_idx)
        routes = bellman_rollout_nexthop(
            raw_model,
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            origins_b,
            dests_b,
            edge_time_matrix,
            max_steps=cfg.max_seq,
        )
        all_routes.extend(routes)
        all_gt.extend(gt_routes)
        all_origins_l.extend(origins_b.tolist())
        all_dests_l.extend(dests_b.tolist())
        for gt in gt_routes:
            strat_keys.append(min(len(r) for r in gt))

    metrics = compute_nexthop_rollout_metrics(
        all_routes,
        torch.tensor(all_dests_l, device=device),
        all_gt,
        edge_time_matrix=edge_time_matrix,
        optimal_times=optimal_times_eval,
        origins=torch.tensor(all_origins_l, device=device),
        strat_keys=strat_keys,
    )
    rprint(f"  {metrics}")

    if metrics.stratified:
        bucket_ranges = [(2, 5), (6, 10), (11, 20), (21, 30), (31, 50)]
        bucket_parts = []
        for lo, hi in bucket_ranges:
            total_n = 0
            total_succ = 0
            dij_vals = []
            for k, (succ, _lr, dij, n) in metrics.stratified.items():
                if lo <= k <= hi:
                    total_n += n
                    total_succ += succ * n
                    if dij != float("inf"):
                        dij_vals.extend([dij] * n)
            if total_n > 0:
                avg_d = sum(dij_vals) / len(dij_vals) if dij_vals else float("inf")
                bucket_parts.append(
                    f"{lo}-{hi}st:{total_succ / total_n:.0%} dij={avg_d:.2f}({total_n})"
                )
        rprint(f"  stratified: {' | '.join(bucket_parts)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_TYPES), default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--value-primary", action="store_true")
    args = p.parse_args()

    cfg = TrainConfig.from_defaults(
        model_type=args.model,
        profile=args.profile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        eval_only=args.eval_only,
        value_primary=args.value_primary or None,
    )
    print(f"Config: {cfg.hp_tag}")
    train(cfg)
