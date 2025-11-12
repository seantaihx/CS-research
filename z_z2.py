#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

# ---------- Configuration ----------
DATA_DIR = Path("./")
OUT_DIR = Path("./results_unify")
OUT_DIR.mkdir(exist_ok=True)

BLOCK_SIZE = 30.0   # seconds per block
L2 = 1e-2           # regularization weight

# ---------- Helpers ----------
def ts_float(x):
    return float(x) if not isinstance(x, (list, tuple)) else float(x[0])

def greedy_mean_decomp(ts, y, tasks):
    """Iterative mean-only greedy separation per node."""
    T = len(ts)
    contribs = {t["task_id"]: np.zeros(T) for t in tasks}
    if not tasks:
        return contribs, y.copy(), np.zeros_like(y)

    tids = [t["task_id"] for t in tasks]
    active = np.vstack([(ts >= t["start"]) & (ts < t["finish"]) for t in tasks])

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


def mean_then_blockwise_nnls(ts, y, tasks, block_size=30.0, l2=1e-2):
    """Blockwise NNLS refinement."""
    T = len(ts)
    contribs = {t["task_id"]: np.zeros(T) for t in tasks}
    if not tasks:
        return contribs

    A_cols, ids = [], []
    for t in tasks:
        start, fin = t["start"], t["finish"]
        nb = max(1, int(np.ceil((fin - start) / block_size)))
        edges = np.linspace(start, fin, nb + 1)
        for b in range(nb):
            m = (ts >= edges[b]) & (ts < edges[b + 1])
            A_cols.append(m.astype(float))
            ids.append((t["task_id"], b))
    if not A_cols:
        return contribs

    A = np.vstack(A_cols).T
    A_aug = np.vstack([A, np.sqrt(l2) * np.eye(A.shape[1])])
    y_aug = np.concatenate([y, np.zeros(A.shape[1])])
    res = lsq_linear(A_aug, y_aug, bounds=(0, np.inf))
    x = res.x

    for k in contribs:
        contribs[k][:] = 0
    for c, (tid, _) in enumerate(ids):
        contribs[tid] += A[:, c] * x[c]
    return contribs


# ---------- Load data ----------
system_all = json.load(open(DATA_DIR / "all_system_loads_ic2.json"))
workloads_all = json.load(open(DATA_DIR / "all_workloads_ic2.json"))
assert len(system_all) == len(workloads_all), "System/workload list lengths differ!"
print(f"Loaded {len(system_all)} workload+system pairs.")

# ---------- Iterate all workloads ----------
for wi, (system_entry, workload_entry) in enumerate(zip(system_all, workloads_all)):
    wname = workload_entry.get("workload_name") or workload_entry.get("name") or f"w{wi}"
    print(f"\n=== Processing {wname} (index {wi}) ===")

    plotdir = OUT_DIR / f"plots_{wi}_{wname.replace(' ', '_')}"
    plotdir.mkdir(parents=True, exist_ok=True)

    # Build node utilization
    node_series = {}
    for node in system_entry.get("node_list", []):
        name = node.get("node_name")
        pairs = node.get("metrics", {}).get("cpu_util", [])
        if not pairs:
            continue
        t, v = zip(*pairs)
        ts = np.array(list(map(float, t)))
        vals = np.array(list(map(float, v)))
        order = np.argsort(ts)
        ts, vals = ts[order], vals[order]
        node_series[name] = {"timestamps": ts, "util": vals}

    # Build task list
    tasks_all = [{
        "task_id": int(t["task_id"]),
        "start": float(t.get("start_time", t.get("submit_time", 0))),
        "finish": float(t.get("finish_time", 0)),
        "nodes": [n["node_name"] for n in t.get("nodes", []) if n.get("node_name")]
    } for t in workload_entry.get("tasklist", [])]

    # Process each node
    for node_name, data in node_series.items():
        ts_abs = data["timestamps"]
        if len(ts_abs) == 0:
            continue
        y = data["util"]

        # Convert timestamps to relative duration (seconds since start)
        ts_rel = ts_abs - ts_abs[0]
        node_tasks = [t for t in tasks_all if node_name in t["nodes"]]

        # Greedy
        contribs_greedy, residual_greedy, recon_greedy = greedy_mean_decomp(ts_rel, y, node_tasks)

        # NNLS
        contribs_nnls = mean_then_blockwise_nnls(ts_rel, y, node_tasks, BLOCK_SIZE, L2)
        recon_nnls = np.sum(np.vstack(list(contribs_nnls.values())), axis=0) if contribs_nnls else np.zeros_like(y)

        # Task intervals for Gantt
        intervals = [(t["task_id"], t["start"] - ts_abs[0], t["finish"] - t["start"]) for t in node_tasks]

        # ---------- Plot ----------
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # (1) Gantt chart
        ax = axes[0]
        if intervals:
            task_ids, starts, durations = zip(*intervals)
            ax.barh(task_ids, durations, left=starts, height=0.6, color="skyblue")
        ax.set_title(f"Tasks on {node_name}")
        ax.set_ylabel("Task ID")
        ax.invert_yaxis()

        # (2) Greedy plot
        ax = axes[1]
        ax.plot(ts_rel, y, label="Observed", lw=1.2)
        ax.plot(ts_rel, recon_greedy, label="Reconstructed", lw=1.0)
        ax.set_title(f"{node_name} — Greedy Mean")
        ax.set_ylabel("Util")
        ax.legend()

        # (3) NNLS plot
        ax = axes[2]
        ax.plot(ts_rel, y, label="Observed", lw=1.2)
        ax.plot(ts_rel, recon_nnls, label="Reconstructed", lw=1.0)
        ax.set_title(f"{node_name} — Blockwise NNLS")
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Util")
        ax.legend()

        fig.tight_layout()
        out_path = plotdir / f"{node_name}.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"  Saved: {out_path.name}")

print("\nDone. Each node plot uses duration (seconds) on the x-axis.")