#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Iterative mean-based task separation algorithm
----------------------------------------------
Given:
 - system_load_scattered.json  : Node-level utilization (5s intervals)
 - workloads_scattered.json    : Task start/end times & CPUs

Steps:
 1. Build per-node utilization time series.
 2. Map tasks to nodes based on their recorded nodes.
 3. For each node:
      - Start with total utilization as residual.
      - Identify intervals where exactly one task is active.
      - Estimate mean utilization of that task from those intervals.
      - Subtract its contribution from residual.
      - Repeat until no more tasks can be estimated.
 4. Save per-node, per-task utilization arrays + residuals.
 5. Export to CSV & PNG plots for verification.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import math

# ==== CONFIG ====

def to_float_ts(x):
    try:
        return float(x)
    except Exception:
        return float(str(x))

# === BUILD NODE UTILIZATION ===

def separate_utilization_per_workload(i): # i is index of the workload
    node_series = {}
    entry = system_data[i]
    workload_name = entry["workload-name"]
    for node in entry.get("node_list", []):
        name = node.get("node_name")
        metrics = node.get("metrics", {})
        cpu_list = metrics.get("cpu_util", [])
        if name not in node_series:
            node_series[name] = {"timestamps": [], "util": []}
        for ts_str, val_str in cpu_list:
            ts = to_float_ts(ts_str)
            val = float(val_str) if val_str not in [None, ""] else np.nan
            node_series[name]["timestamps"].append(ts)
            node_series[name]["util"].append(val)

    for name, d in node_series.items():
        arr = np.array(d["timestamps"], dtype=float)
        vals = np.array(d["util"], dtype=float)
        order = np.argsort(arr)
        arr, vals = arr[order], vals[order]
        uniq_ts, inv = np.unique(arr, return_inverse=True)
        if len(uniq_ts) < len(arr):
            avg_vals = np.zeros(len(uniq_ts))
            counts = np.zeros(len(uniq_ts))
            for i, idx in enumerate(inv):
                avg_vals[idx] += vals[i]
                counts[idx] += 1
            vals = avg_vals / counts
        node_series[name]["timestamps"] = uniq_ts
        node_series[name]["util"] = vals

# === BUILD TASK TABLE ===
    workload_entry = workloads_data[i]
    tasklist = workload_entry.get("tasklist", [])

    tasks = []
    for t in tasklist:
        task_id = int(t.get("task_id"))
        start = to_float_ts(t.get("start_time", t.get("submit_time")))
        finish = to_float_ts(t.get("finish_time", start + 1e-6))
        cpus = float(t.get("cpus", 0.0))
        node_names = [nd.get("node_name") for nd in t.get("nodes", []) if nd.get("node_name")]
        tasks.append({
            "task_id": task_id,
            "start": start,
            "finish": finish,
            "cpus_total": cpus,
            "nodes": list(set(node_names)),
        })

    node_to_tasks = {n: [] for n in node_series.keys()}
    for i, t in enumerate(tasks):
        for n in t["nodes"]:
            if n in node_to_tasks:
                node_to_tasks[n].append(i)

    # === ITERATIVE MEAN SUBTRACTION ===
    per_node_timestamps = {n: np.array(d["timestamps"]) for n, d in node_series.items()}
    per_node_util = {n: np.array(d["util"]) for n, d in node_series.items()}
    per_node_task_contribs = {}
    per_node_residuals = {}
    summary_rows = []

    print("Running iterative mean-based separation...")
    #print(per_node_timestamps)
    for node_name, ts in per_node_timestamps.items():
        y = per_node_util[node_name].astype(float).copy() # time stamps
        T = len(ts) # number of time stamps
        task_indices = node_to_tasks.get(node_name, []) # task indices on that node
        #print(task_indices)
        Ntasks = len(task_indices) # number of tasks
        if Ntasks == 0:
            per_node_task_contribs[node_name] = {}
            per_node_residuals[node_name] = y
            continue

        active = np.zeros((Ntasks, T), dtype=bool)
        #print(ts)
        for j, ti in enumerate(task_indices):
            tinfo = tasks[ti]
            active[j] = (ts >= tinfo["start"]) & (ts < tinfo["finish"])
            #print(active[j], end = '\n\n\n\n\n')
        residual = y.copy()
        contribs = np.zeros((Ntasks, T))
        known = np.zeros(Ntasks, dtype=bool)
        mean_est = np.full(Ntasks, np.nan)

        for i in range(40):
            progress = False
            active_sum = active.sum(axis=0) # number of true values on axis 0
            print(active_sum, end = "\n\n")
            for j in range(Ntasks):
                if known[j]:
                    continue

                mask_only = active[j] & (active_sum == 1)
                #print(mask_only, end = "\n\n")
                if not mask_only.any():
                    mask_ok = active[j].copy()
                    for k in range(Ntasks):
                        if k == j or known[k]:
                            continue
                        mask_ok &= ~active[k]
                    mask_candidate = mask_ok
                else:
                    mask_candidate = mask_only

                if mask_candidate.any():
                    est = np.nanmean(residual[mask_candidate])
                    est = max(0.0, est)
                    mean_est[j] = est
                    contribs[j] = active[j].astype(float) * est
                    residual -= contribs[j]
                    known[j] = True
                    progress = True
            if not progress:
                break

        per_node_task_contribs[node_name] = {}
        for j, ti in enumerate(task_indices):
            tid = tasks[ti]["task_id"]
            per_node_task_contribs[node_name][tid] = contribs[j]
            summary_rows.append({
                "node": node_name,
                "task_id": tid,
                "mean_est": float(mean_est[j]) if not math.isnan(mean_est[j]) else None,
            })

        per_node_residuals[node_name] = residual

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"per_node_task_summary_{workload_name}_{i}.csv", index=False)
    print("Separation complete.")

    # === SAVE & PLOT ===
    np.save(out_dir / f"per_node_task_contribs_{workload_name}_{i}.npy", per_node_task_contribs, allow_pickle=True)
    np.save(out_dir / f"per_node_timestamps_{workload_name}_{i}.npy", per_node_timestamps, allow_pickle=True)
    np.save(out_dir / f"per_node_residuals_{workload_name}{i}.npy", per_node_residuals, allow_pickle=True)

    print("Generating plots...")
    for node, ts in per_node_timestamps.items():
        contrib_sum = sum(per_node_task_contribs[node].values())
        residual = per_node_residuals[node]
        y_obs = contrib_sum + residual
        plt.figure(figsize=(10, 3))
        plt.plot(ts, y_obs, label="Observed")
        plt.plot(ts, contrib_sum, label="Reconstructed")
        #plt.plot(ts, residual, label="Residual")
        plt.title(f"Node {node}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"plot_{node}_{workload_name}_{i}.png")
        plt.close()

    print(f"✅ Results saved under: {out_dir.resolve()}")
    
    multi_plot_dir = out_dir / "plots_all_tasks"
    multi_plot_dir.mkdir(exist_ok=True)

    print("Generating per-node all-task utilization plots...")
    for node, ts in per_node_timestamps.items():
        contribs = per_node_task_contribs[node]
        if not contribs:
            continue

        plt.figure(figsize=(10, 4))
        # Plot each task separately
        for tid, arr in contribs.items():
            plt.plot(ts, arr, label=f"Task {tid}", alpha=0.8)

        # Also plot the node total
        #total = np.sum(list(contribs.values()), axis=0)
        #plt.plot(ts, total, color="black", linewidth=2, label="Sum of tasks")
        
        plt.title(f"Node {node} — per-task estimated utilization")
        plt.xlabel("Timestamp (s)")
        plt.ylabel("CPU Utilization")
        plt.legend(loc="upper right", ncol=2, fontsize=8)
        plt.tight_layout()
        out_file = multi_plot_dir / f"node_{node}_{workload_name}_all_tasks.png"
        plt.savefig(out_file)
        plt.close()

    print(f"✅ Multi-task plots saved under: {multi_plot_dir.resolve()}")

if __name__ == "__main__":
    system_file = Path("all_system_loads_ic2.json")
    workload_file = Path("all_workloads_ic2.json")
    out_dir = Path("./results_greedy_mean")
    out_dir.mkdir(exist_ok=True)

    print("Loading system utilization data...")
    with system_file.open() as f:
        system_data = json.load(f)

    print("Loading workload metadata...")
    with workload_file.open() as f:
        workloads_data = json.load(f)
    
    for index in range(len(system_data)):
        separate_utilization_per_workload(index)
        
