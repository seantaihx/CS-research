#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

# ---------- Configuration ----------
DATA_DIR = Path("./")

# choose ONE workload index to run
i = 15  # <<< change this to the workload you want

BLOCK_SIZE = 30.0   # seconds per block
L2 = 1e-2           # regularization weight

# ---------- Helpers ----------
def ts_float(x): 
    return float(x) if not isinstance(x, (list, tuple)) else float(x[0])

# ---------- Greedy mean-based (iterative) ----------
def greedy_mean_decomp(ts, y, tasks):
    """
    Iterative mean-only greedy separation (per node).
    Returns: contribs dict (task_id -> (T,) array), residual (T,), recon (T,)
    """
    T = len(ts)
    contribs = {t["task_id"]: np.zeros(T) for t in tasks}
    if not tasks:
        return contribs, y.copy(), np.zeros_like(y)

    # activity masks per task
    active = []
    tids = []
    for t in tasks:
        tids.append(t["task_id"])
        mask = (ts >= t["start"]) & (ts < t["finish"])
        active.append(mask)
    active = np.vstack(active) if active else np.zeros((0, T), dtype=bool)

    residual = y.copy().astype(float)
    known = np.zeros(len(tasks), dtype=bool)
    # iterate up to a small cap (like the version you shared)
    for _ in range(40):
        progress = False
        active_sum = active.sum(axis=0)
        for j in range(len(tasks)):
            if known[j]:
                continue
            # use intervals where only this task is active; else try excluding unknown others
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

# ---------- Blockwise NNLS (unchanged calculations) ----------
def mean_then_blockwise_nnls(ts, y, tasks, block_size=30, l2=1e-2):
    T, N = len(ts), len(tasks)
    contribs = {t["task_id"]: np.zeros(T) for t in tasks}

    # --- Blockwise NNLS refinement ---
    A_cols, ids = [], []
    for t in tasks:
        start, fin = t["start"], t["finish"]
        nb = max(1, int(np.ceil((fin - start) / block_size)))
        edges = np.linspace(start, fin, nb + 1)
        for b in range(nb):
            m = (ts >= edges[b]) & (ts < edges[b + 1])
            A_cols.append(m.astype(float)); ids.append((t["task_id"], b))
    if not A_cols:
        return contribs

    A = np.vstack(A_cols).T
    A_aug = np.vstack([A, np.sqrt(l2) * np.eye(A.shape[1])])
    y_aug = np.concatenate([y, np.zeros(A.shape[1])])
    res = lsq_linear(A_aug, y_aug, bounds=(0, np.inf)); x = res.x

    for k in contribs: 
        contribs[k][:] = 0
    for c, (tid, _) in enumerate(ids): 
        contribs[tid] += A[:, c] * x[c]
    return contribs

# ---------- Load data ----------
system_all = json.load(open(DATA_DIR/"all_system_loads_ic2.json"))
workloads_all = json.load(open(DATA_DIR/"all_workloads_ic2.json"))
assert len(system_all) == len(workloads_all), "System/workload list lengths differ!"

# ---------- Pick ONE workload ----------
system_entry = system_all[i]
workload_entry = workloads_all[i]
wname = workload_entry.get("workload_name") or workload_entry.get("name") or f"w{i}"
print(f"\n=== Processing {wname} (pair {i}) ===")

# Build node utilization for this workload
node_series = {}
for node in system_entry.get("node_list", []):
    name = node.get("node_name")
    pairs = node.get("metrics", {}).get("memory_util", [])
    if not pairs: 
        continue
    t, v = zip(*pairs)
    ts = np.array(list(map(float, t)))
    vals = np.array(list(map(float, v)))
    order = np.argsort(ts)
    node_series[name] = {
        "timestamps": ts[order],
        "util": vals[order]
    }

# Build task list for this workload
tasks_all = [{
    "task_id": int(t["task_id"]),
    "start": float(t.get("start_time", t.get("submit_time", 0))),
    "finish": float(t.get("finish_time", 0)),
    "nodes": [n["node_name"] for n in t.get("nodes", []) if n.get("node_name")]
} for t in workload_entry.get("tasklist", [])]

# ---------- One figure (3 subplots) per node ----------
for node_name, data in node_series.items():
    ts = data["timestamps"]; y = data["util"]
    # tasks assigned to this node
    node_tasks = [t for t in tasks_all if node_name in t["nodes"]]

    # 1) intervals for Gantt-like bars
    node_to_intervals = {}
    for tsk in node_tasks:
        start_time = float(tsk["start"])
        finish_time = float(tsk["finish"])
        task_id = tsk["task_id"]
        duration = finish_time - start_time
        node_to_intervals.setdefault(node_name, []).append((task_id, start_time, duration))
    intervals = node_to_intervals.get(node_name, [])

    # 2) Greedy mean-based
    contribs_greedy, residual_greedy, recon_greedy = greedy_mean_decomp(ts, y, node_tasks)

    # 3) NNLS
    contribs_nnls = mean_then_blockwise_nnls(ts, y, node_tasks, BLOCK_SIZE, L2)
    recon_nnls = np.sum(np.vstack(list(contribs_nnls.values())), axis=0) if contribs_nnls else np.zeros_like(y)

    # ---------- Plot: 3 subplots ----------
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Subplot 1: Gantt-style bars
    ax = axes[0]
    if intervals:
        task_ids, starts, durations = zip(*intervals)
        ax.barh(task_ids, durations, left=starts, height=0.6, color='skyblue')
    ax.set_title(f"Tasks on {node_name}")
    ax.set_ylabel("Task ID")
    ax.invert_yaxis()

    # Subplot 2: Greedy (Observed vs Reconstructed vs Residual)
    ax = axes[1]
    ax.plot(ts, y, label="Observed", lw=1.2)
    ax.plot(ts, recon_greedy, label="Reconstructed", lw=1.0)
    ax.plot(ts, residual_greedy, label="Residual", lw=1.0)
    ax.set_title(f"Node {node_name} — Greedy")
    ax.set_ylabel("Util")
    ax.legend(loc="best")

    # Subplot 3: NNLS (Observed vs Reconstructed) — like your 3rd pic
    ax = axes[2]
    ax.plot(ts, y, label="Observed", lw=1.2)
    ax.plot(ts, recon_nnls, label="Reconstructed", lw=1.0)
    ax.set_title(f"Node {node_name} — NNLS")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Util")
    ax.legend(loc="best")

    fig.tight_layout()
    # one PNG per node, 3 subplots each
    plt.savefig(f"workload[{i}]_node_{node_name}.png", dpi=300)
    plt.close(fig)

print("Done. Saved one PNG per node with 3 subplots (Gantt, Greedy, NNLS).")