#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

# ==================== Config ====================
DATA_DIR = Path("./")
OUT_DIR  = Path("./ic2_memory")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE = 30.0   # seconds per block (for NNLS)
L2         = 1e-2   # L2 regularization (for NNLS)

# ==================== Helpers ====================
def _safe_name(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (s or ""))

def _dedup_avg(ts, vals):
    """Average duplicate timestamps; return (uniq_ts, averaged_vals)."""
    ts = np.asarray(ts, float); vals = np.asarray(vals, float)
    order = np.argsort(ts); ts = ts[order]; vals = vals[order]
    uniq, inv = np.unique(ts, return_inverse=True)
    if len(uniq) == len(ts):
        return ts, vals
    acc = np.zeros(len(uniq), float)
    cnt = np.zeros(len(uniq), float)
    for i, k in enumerate(inv):
        acc[k] += vals[i]; cnt[k] += 1.0
    return uniq, acc / np.maximum(cnt, 1.0)

def _auto_time_scale(max_time_value):
    """If timestamps look like milliseconds since epoch, scale to seconds."""
    # Heuristic: >1e12 ~ ms epoch; 1e9..1e12 ~ s epoch; <=1e9 often relative seconds
    return 1.0/1000.0 if max_time_value > 1e12 else 1.0

def _clip_interval(a, b, lo, hi):
    """Return intersection [max(a,lo), min(b,hi)], or (None,None) if empty."""
    s = max(a, lo); e = min(b, hi)
    return (s, e) if e > s else (None, None)

# ==================== Greedy mean (unchanged math) ====================
def greedy_mean_decomp(ts, y, tasks):
    """
    Iterative mean-only greedy separation (per node).
    Returns: contribs dict (task_id -> (T,) array), residual (T,), recon (T,)
    """
    T = len(ts)
    contribs = {t["task_id"]: np.zeros(T) for t in tasks}
    if not tasks:
        return contribs, y.copy(), np.zeros_like(y)

    tids   = [t["task_id"] for t in tasks]
    active = np.vstack([(ts >= t["start"]) & (ts < t["finish"]) for t in tasks]) if tasks else np.zeros((0,T),bool)

    residual = y.copy().astype(float)
    known = np.zeros(len(tasks), dtype=bool)

    for _ in range(40):
        progress = False
        active_sum = active.sum(axis=0)
        for j in range(len(tasks)):
            if known[j]:
                continue
            mask_only = active[j] & (active_sum == 1)
            if not mask_only.any():
                mask_ok = active[j].copy()
                for k in range(len(tasks)):
                    if k == j or known[k]:
                        continue
                    mask_ok &= ~active[k]
                mask = mask_ok
            else:
                mask = mask_only

            if mask.any():
                est = np.nanmean(residual[mask])
                est = max(0.0, est)
                contrib = active[j].astype(float) * est
                contribs[tids[j]] = contrib
                residual -= contrib
                known[j] = True
                progress = True
        if not progress:
            break

    recon = np.sum(np.vstack(list(contribs.values())), axis=0) if contribs else np.zeros_like(y)
    return contribs, residual, recon

# ==================== Blockwise NNLS (unchanged math; returns recon too) ====================
def mean_then_blockwise_nnls(ts, y, tasks, block_size=30.0, l2=1e-2):
    """
    Blockwise regularized NNLS refinement (same calculations).
    Returns: contribs dict (task_id -> (T,) array), recon (T,)
    """
    T = len(ts)
    contribs = {t["task_id"]: np.zeros(T) for t in tasks}
    if not tasks:
        return contribs, np.zeros_like(y)

    A_cols, ids = [], []
    for t in tasks:
        start, fin = float(t["start"]), float(t["finish"])
        if fin <= start:
            continue
        nb = max(1, int(np.ceil((fin - start) / block_size)))
        edges = np.linspace(start, fin, nb + 1)
        for b in range(nb):
            m = (ts >= edges[b]) & (ts < edges[b + 1])
            if not m.any():
                continue
            A_cols.append(m.astype(float))
            ids.append((t["task_id"], b))

    if not A_cols:
        return contribs, np.zeros_like(y)

    A = np.vstack(A_cols).T  # shape (T, K)
    A_aug = np.vstack([A, np.sqrt(l2) * np.eye(A.shape[1])])
    y_aug = np.concatenate([y, np.zeros(A.shape[1])])
    res = lsq_linear(A_aug, y_aug, bounds=(0, np.inf))
    x = res.x  # shape (K,)

    recon = A @ x
    for k in contribs:
        contribs[k][:] = 0.0
    for c, (tid, _) in enumerate(ids):
        contribs[tid] += A[:, c] * x[c]
    return contribs, recon

# ==================== Load data ====================
system_all    = json.load(open(DATA_DIR / "all_system_loads_ic2.json"))
workloads_all = json.load(open(DATA_DIR / "all_workloads_ic2.json"))
assert len(system_all) == len(workloads_all), "System/workload list lengths differ!"

print(f"Loaded {len(system_all)} workload+system pairs.")

# ==================== Iterate all workloads ====================
for wi, (system_entry, workload_entry) in enumerate(zip(system_all, workloads_all), start=0):
    wname = workload_entry.get("workload_name") or workload_entry.get("name") or f"w{wi}"
    safe = _safe_name(wname)
    plotdir = OUT_DIR / f"{wi:03d}_{safe}"
    plotdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Processing [{wi}] {wname} ===")

    # -------- Build node utilization (with de-dup + sorting) --------
    node_series = {}
    node_max_time = 0.0
    for node in system_entry.get("node_list", []):
        name  = node.get("node_name")
        pairs = node.get("metrics", {}).get("memory_util", [])
        if not pairs:
            continue
        t, v = zip(*pairs)
        ts   = np.array(list(map(float, t)))
        vals = np.array(list(map(float, v)))
        ts, vals = _dedup_avg(ts, vals)
        node_series[name] = {"timestamps": ts, "util": vals}
        node_max_time = max(node_max_time, float(ts.max()) if len(ts) else 0.0)

    # -------- Build raw task list --------
    raw_tasks = [{
        "task_id": int(t["task_id"]),
        "start": float(t.get("start_time", t.get("submit_time", 0))),
        "finish": float(t.get("finish_time", 0)),
        "nodes": [n["node_name"] for n in t.get("nodes", []) if n.get("node_name")]
    } for t in workload_entry.get("tasklist", [])]

    # -------- Normalize time units consistently (seconds) --------
    # Decide scale using the largest node timestamp and task time.
    task_max_time = 0.0
    if raw_tasks:
        task_max_time = max(max(t["start"], t["finish"]) for t in raw_tasks)
    scale = _auto_time_scale(max(node_max_time, task_max_time))

    # Scale node times
    for node_name in list(node_series.keys()):
        ts = node_series[node_name]["timestamps"] * scale
        node_series[node_name]["timestamps"] = ts
        # util unchanged

    # Scale task times
    for t in raw_tasks:
        t["start"]  *= scale
        t["finish"] *= scale

    # -------- Per-node processing --------
    for node_name, data in node_series.items():
        ts_abs = data["timestamps"]
        y      = data["util"]
        if len(ts_abs) == 0:
            continue

        t0, t1 = float(ts_abs[0]), float(ts_abs[-1])

        # Keep only tasks that intersect this node time window
        node_tasks = []
        for t in raw_tasks:
            if node_name not in t["nodes"]:
                continue
            s, e = float(t["start"]), float(t["finish"])
            cs, ce = _clip_interval(s, e, t0, t1)
            if cs is None:
                continue
            node_tasks.append({"task_id": t["task_id"], "start": cs, "finish": ce, "nodes": t["nodes"]})

        # --- Greedy & NNLS on aligned absolute seconds ---
        contribs_greedy, residual_greedy, recon_greedy = greedy_mean_decomp(ts_abs, y, node_tasks)
        contribs_nnls,   recon_nnls                   = mean_then_blockwise_nnls(ts_abs, y, node_tasks, BLOCK_SIZE, L2)

        # --- Plot x-axis as elapsed seconds since node start (no math change) ---
        ts_plot = ts_abs - t0

        # Gantt bars clipped to [t0, t1]
        intervals = []
        for t in node_tasks:
            left  = float(t["start"] - t0)
            width = float(t["finish"] - t["start"])
            if width > 0:
                intervals.append((t["task_id"], left, width))

        # -------------------- Plot (3 subplots) --------------------
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # (1) Gantt
        ax = axes[0]
        if intervals:
            task_ids, starts, durations = zip(*intervals)
            ax.barh(task_ids, durations, left=starts, height=0.6)
        ax.set_title(f"Tasks on {node_name}")
        ax.set_ylabel("Task ID")
        ax.invert_yaxis()
        

        # (2) Greedy — observed vs reconstructed
        ax = axes[1]
        ax.plot(ts_plot, y, label="Observed", lw=1.2)
        ax.plot(ts_plot, recon_greedy, label="Reconstructed (Greedy)", lw=1.0)
        ax.set_title(f"{node_name} — Greedy Mean")
        ax.set_ylabel("Util")
        ax.legend(loc="best")
        

        # (3) Blockwise NNLS — observed vs reconstructed
        ax = axes[2]
        ax.plot(ts_plot, y, label="Observed", lw=1.2)
        ax.plot(ts_plot, recon_nnls, label="Reconstructed (NNLS)", lw=1.0)
        ax.set_title(f"{node_name} — Blockwise NNLS (block={BLOCK_SIZE}s, L2={L2})")
        ax.set_xlabel("Time since node start (s)")
        ax.set_ylabel("Util")
        ax.legend(loc="best")

        fig.tight_layout()
        out_png = plotdir / f"{node_name}.png"
        plt.savefig(out_png, dpi=220)
        plt.close(fig)
        print(f"  - Saved {out_png}")