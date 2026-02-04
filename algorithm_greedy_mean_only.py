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


def pertask_utilization_greedy():

    system_all = json.load(open("all_system_loads_ic2.json", "r"))
    workloads_all = json.load(open("all_workloads_ic2.json", "r"))
    assert len(system_all) == len(workloads_all)

    contribs_all = {}

    for wi, (system_entry, workload_entry) in enumerate(zip(system_all, workloads_all)):
        wname = workload_entry.get("workload_name", f"w{wi}")
        #print(wname)

        node_series = {}
        for node in system_entry.get("node_list", []):
            name = node.get("node_name")
            #print(name)
            pairs = node.get("metrics", {}).get("cpu_util", [])
            if (not name) or (not pairs):
                continue

            t, v = zip(*pairs)  # list of (timestamp, util)
            ts = np.array(list(map(float, t)))
            y = np.array(list(map(float, v)))

            order = np.argsort(ts)
            node_series[name] = {
                "timestamps": ts[order],
                "util": y[order],
            }

        if not node_series:
            continue


        tasks = []
        for t in workload_entry.get("tasklist", []):
            tid = t.get("task_id", t.get("id"))
            if tid is None:
                continue

            start = float(t.get("start_time", t.get("submit_time", 0.0)))
            finish = float(t.get("finish_time", start + 1e-6))


            nodes = t.get("nodes", []) #get node field or empty list if missing
            node_names = []
            for n in nodes:
                nn = n.get("node_name")
                node_names.append(nn)
                #print(nn)


            tasks.append({
                "task_id": tid,
                "start": start,
                "finish": finish,
                "nodes": node_names,
            })
        if not tasks:
            continue


        node_to_tasks = {n: [] for n in node_series.keys()}
        for i, t in enumerate(tasks):
            for n in t["nodes"]:
                if n in node_to_tasks:
                    node_to_tasks[n].append(i)


        for node, data in node_series.items():
            task_indices = node_to_tasks.get(node, [])
            if not task_indices:
                continue

            ts = data["timestamps"] #per node
            y = data["util"]    #per node

            T = len(ts)
            Ntasks = len(task_indices)

            # active mask per task-on-this-node
            active = np.zeros((Ntasks, T), dtype=bool)
            tids = []
            for j, ti in enumerate(task_indices):
                tinfo = tasks[ti]
                tids.append(tinfo["task_id"])
                active[j] = (ts >= tinfo["start"]) & (ts < tinfo["finish"])


            residual = y.copy()
            known = np.zeros(Ntasks, dtype=bool)
            ests = np.zeros(Ntasks)
            mean_est = np.full(Ntasks, np.nan)
            contribs = np.zeros((Ntasks, T))

            for iters in range(40):
                best_j = None
                best_mse = None
                best_est = None

                for j in range(Ntasks):
                    if known[j]:
                        continue
                    mask_only = active[j]
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
                        est = float(np.mean(residual[mask_candidate]))
                        est = max(0.0, est)
                        mean_est[j] = est
                        contribs[j] = active[j].astype(float) * est
                        residual -= contribs[j]
                        known[j] = True
                        progress = True

                    if not progress:
                        break



            active_contribs = []
            for j in range(Ntasks):
                util = active[j].astype(float) * mean_est[j]
                mask = util > 0.0
                if not np.any(mask):
                    continue

                active_contribs.append({
                    "task_id": tids[j],
                    "Node_name": node,
                    "Util": list(zip(ts[mask].tolist(), util[mask].tolist())),
                })

            if active_contribs:
                contribs_all[(wname, node)] = active_contribs
                #print(node)
            #print(contribs_all)

    return contribs_all
 
import numpy as np

def greedy_mean_separation(ts, y, tasks, *, max_iters=40, eps=1e-12):
    """
    Greedy mean separation (your PDF logic, but cleaned up).

    Inputs
    - ts: 1D array of timestamps (sorted or unsorted)
    - y:  1D array of cpu util (same length as ts)
    - tasks: list of dicts, each must have:
        {"task_id": ..., "start": ..., "finish": ...}

    Returns
    - contribs_by_tid: dict {task_id: contrib_vector}
    - recon: 1D array, sum of all contribs
    - residual: 1D array, y - recon
    - mean_est_by_tid: dict {task_id: estimated_mean (float or None)}
    """

    ts = np.asarray(ts, dtype=float)
    y = np.asarray(y, dtype=float)
    assert ts.shape == y.shape and ts.ndim == 1

    # Ensure sorted by time (important for consistent masks/plots)
    order = np.argsort(ts)
    ts = ts[order]
    y = y[order]

    T = len(ts)
    N = len(tasks)

    # Build active masks: active[j, t] = True if task j active at timestamp t
    active = np.zeros((N, T), dtype=bool)
    task_ids = []
    for j, tsk in enumerate(tasks):
        tid = tsk["task_id"]
        start = float(tsk["start"])
        finish = float(tsk["finish"])
        task_ids.append(tid)

        # half-open interval [start, finish)
        active[j] = (ts >= start) & (ts < finish)

    residual = y.copy()
    known = np.zeros(N, dtype=bool)
    mean_est = np.full(N, np.nan)
    contribs = np.zeros((N, T), dtype=float)

    for _ in range(max_iters):
        progress = False
        active_sum = active.sum(axis=0)  # how many tasks active at each timestamp

        for j in range(N):
            if known[j]:
                continue

            # 1) best case: timestamps where ONLY this task is active
            mask_only = active[j] & (active_sum == 1)

            if mask_only.any():
                mask_candidate = mask_only
            else:
                # 2) fallback: task j active, and no OTHER UNKNOWN tasks active
                mask_ok = active[j].copy()
                for k in range(N):
                    if k == j or known[k]:
                        continue
                    mask_ok &= ~active[k]
                mask_candidate = mask_ok

            if mask_candidate.any():
                est = np.nanmean(residual[mask_candidate])
                if not np.isfinite(est):
                    continue

                est = max(0.0, float(est))  # keep non-negative like your code
                mean_est[j] = est

                cj = active[j].astype(float) * est
                contribs[j] = cj
                residual = residual - cj

                known[j] = True
                progress = True

        if not progress:
            break

    # Pack results
    contribs_by_tid = {task_ids[j]: contribs[j] for j in range(N)}
    recon = contribs.sum(axis=0)
    residual = y - recon  # safer than accumulated subtraction drift

    mean_est_by_tid = {
        task_ids[j]: (None if np.isnan(mean_est[j]) else float(mean_est[j]))
        for j in range(N)
    }

    return recon

def separate_utilization_per_workload(workload_i, system_data, workloads_data, utilization): # i is index of the workload
    node_series = {}
    entry = system_data[workload_i]
    workload_name = entry["workload-name"]
    for node in entry.get("node_list", []):
        name = node.get("node_name")
        metrics = node.get("metrics", {})
        utilization_list = metrics.get(f"{utilization}_util", [])
        if name not in node_series:
            node_series[name] = {"timestamps": [], "util": []}
        for ts_str, val_str in utilization_list:
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
    workload_entry = workloads_data[workload_i]
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

    #print("Running iterative mean-based separation...")
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

        for iter in range(40):
            progress = False
            active_sum = active.sum(axis=0) # number of true values on axis 0
            #print(active_sum, end = "\n\n")
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
                #print("known",known[j])
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

    #for i in range(len(known)):
        #print("known", known[j], tasks[j]["start"], tasks[j]["finish"])
    '''
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"per_node_task_summary_{workload_name}_{i}.csv", index=False)
    #print("Separation complete.")

    # === SAVE & PLOT ===
    np.save(out_dir / f"per_node_task_contribs_{workload_name}_{i}.npy", per_node_task_contribs, allow_pickle=True)
    np.save(out_dir / f"per_node_timestamps_{workload_name}_{i}.npy", per_node_timestamps, allow_pickle=True)
    np.save(out_dir / f"per_node_residuals_{workload_name}{i}.npy", per_node_residuals, allow_pickle=True)
    '''
    totalMSE = 0
    totalMAPE = 0
    count = 0
    totalMAE = 0
    #print("Generating plots...")
    MAE_list = []
    for node, ts in per_node_timestamps.items():
        node_abs_sum = 0.0
        t_count = 0
        contrib_sum = sum(per_node_task_contribs[node].values())
        residual = per_node_residuals[node]
        y_obs = contrib_sum + residual

        start_finish = set()
        for ti in node_to_tasks.get(node, []):
            s = int(tasks[ti]["start"])
            f = int(tasks[ti]["finish"])
            start_finish.update([s-1, s, s+1, f-1, f, f+1])

        for i2 in range(len(residual)):
            if int(ts[i2]) in start_finish:
                continue
            totalMSE += residual[i2]**2
            totalMAE += abs(residual[i2])
            node_abs_sum += abs(residual[i2])
            t_count += 1
            if y_obs[i2] == 0.0 or residual[i2] < 0:
                continue
            totalMAPE += (abs(residual[i2])/y_obs[i2])*100
            count += 1
        if t_count > 0:
            #print(t_count)
            MAE_list.append(float(node_abs_sum/t_count))

        '''
        plt.figure(figsize=(10, 3))
        plt.plot(ts, y_obs, label="Observed")
        plt.plot(ts, contrib_sum, label="Reconstructed")
        #plt.plot(ts, residual, label="Residual")
        plt.title(f"Node {node}")
        plt.legend()
        plt.tight_layout()
        #plt.savefig(out_dir / f"plot_workload{workload_i}_{node}.png")
        plt.close()
        '''
    MSE = totalMSE/count
    MAPE = totalMAPE/count
    MAE = totalMAE/count

    #print(f"✅ Results saved under: {out_dir.resolve()}")
    
    #multi_plot_dir = out_dir / "plots_all_tasks"
    #multi_plot_dir.mkdir(exist_ok=True)

    #print("Generating per-node all-task utilization plots...")
    '''
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
        #plt.savefig(out_file)
        plt.close()

    #print(f"✅ Multi-task plots saved under: {multi_plot_dir.resolve()}")
    '''
    return MSE, MAPE, MAE, MAE_list

def gm_main(workload_data, system_data, utilization):
    #system_file = Path(system)
    #workload_file = Path(workload)
    #out_dir = Path("./results_greedy_mean")
    #out_dir.mkdir(exist_ok=True)


    totalMSE = 0
    totalMAPE = 0
    totalMAE = 0
    results = []
    all_MAE_list = []
    for index in range(len(system_data)):
        singleMSE, singleMAPE, singleMAE, MAE_list = separate_utilization_per_workload(index, system_data, workload_data, utilization)
        #print(f"Workload {index}: MSE={singleMSE}, MAE={singleMAE}%")
        #print(f"MAE: {singleMAE}")
        totalMSE += singleMSE
        totalMAPE += singleMAPE
        totalMAE += singleMAE
        all_MAE_list.extend(MAE_list)
        #MAE_list.append(float(singleMAE))

    #print(f"GM_{utilization}: {len(all_MAE_list)}")
    
    #print(f"MSE = {totalMSE/len(system_data)}")
    #print(f"MAPE = {totalMAPE/len(system_data)}")
    #print(f"MAE = {totalMAE/len(system_data)}")
    return all_MAE_list

#pertask_utilization_greedy_mean()