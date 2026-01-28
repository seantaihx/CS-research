import json
import numpy as np
import matplotlib.pyplot as plt

from algorithm_blockwise_nonnegative_least_squares import mean_then_blockwise_nnls
from algorithm_greedy_mean_only import greedy_mean_separation


def task_durations_data(workload_file, wid):
    with open(workload_file, "r") as f:
        workloads = json.load(f)

    tasks = workloads[wid]["tasklist"]

    # build node -> [(task_id, start, duration), ...]
    node_to_intervals = {}
    node_min_start = {}

    for task in tasks:
        start_time = float(task["start_time"])
        finish_time = float(task["finish_time"])
        task_id = int(task["task_id"])
        duration = finish_time - start_time

        for node in task.get("nodes", []):
            node_name = node.get("node_name")
            if node_name is None:
                continue

            if node_name not in node_min_start:
                node_min_start[node_name] = start_time
            else:
                node_min_start[node_name] = min(node_min_start[node_name], start_time)

            node_to_intervals.setdefault(node_name, []).append((task_id, start_time, duration))

    # normalize per node (LOCAL min start)
    normalized = {}
    for node, intervals in node_to_intervals.items():
        min_start = node_min_start[node]
        normalized[node] = []
        for tid, start, dur in intervals:
            normalized[node].append((tid, start - min_start, dur))

        # optional: sort by normalized start so bars look nice
        normalized[node].sort(key=lambda x: x[1])

    return normalized


def all_util_data(workload_file, system_file, wid):
    with open(workload_file, "r") as f1:
        workloads = json.load(f1)
    with open(system_file, "r") as f2:
        system_loads = json.load(f2)

    workload = workloads[wid]
    system_load = system_loads[wid]

    # node_series: node -> {"timestamps": np.array([...]), "util": np.array([...])}
    node_series = {}
    for node in system_load.get("node_list", []):
        name = node.get("node_name")
        pairs = node.get("metrics", {}).get("cpu_util", [])  # Polaris CPU util
        if not name or not pairs:
            continue

        t, v = zip(*pairs)
        ts = np.array(list(map(float, t)))
        vals = np.array(list(map(float, v)))

        order = np.argsort(ts)
        node_series[name] = {
            "timestamps": ts[order],
            "util": vals[order],
        }

    # tasks list (node membership)
    tasks = [{
        "task_id": int(t["task_id"]),
        "start": float(t["start_time"]),
        "finish": float(t["finish_time"]),
        "nodes": [n.get("node_name") for n in t.get("nodes", []) if n.get("node_name")],
    } for t in workload.get("tasklist", [])]

    # recon per node (keep original timestamps for algorithms)
    recon_gm_all = {}
    recon_nnls_all = {}

    for node_name, data in node_series.items():
        node_task = [t for t in tasks if node_name in t["nodes"]]
        if not node_task:
            continue

        contribs_nnls = mean_then_blockwise_nnls(
            data["timestamps"], data["util"], node_task, 30.0
        )
        recon_nnls = sum(contribs_nnls.values())

        recon_gm = greedy_mean_separation(
            data["timestamps"], data["util"], node_task
        )

        recon_nnls_all[node_name] = recon_nnls
        recon_gm_all[node_name] = recon_gm

    # normalize timestamps per node at the end (LOCAL min per node)
    node_series_norm = {}
    for node_name, data in node_series.items():
        ts = data["timestamps"]
        t0 = ts.min()
        node_series_norm[node_name] = {
            "timestamps": ts - t0,
            "util": data["util"],
        }

    return node_series_norm, recon_nnls_all, recon_gm_all


def node_sort_key(name: str):
    # tries to sort by trailing digits (e.g., "node1", "polaris-10", etc.)
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 10**9


if __name__ == "__main__":
    # Polaris files (change names to your actual polaris json filenames)
    workload_file = "all_workloads_polaris.json"
    system_file = "all_system_loads_polaris.json"
    wid = 17  # change to workload index you want

    normalized_task_durations = task_durations_data(workload_file, wid)
    node_series_norm, recon_nnls_all, recon_gm_all = all_util_data(workload_file, system_file, wid)


    nodes = list(node_series_norm.keys())[:3]
    print(nodes)

    fig, axes = plt.subplots(len(nodes), 2, figsize=(18, 2.2 * len(nodes)), sharex=False)
    fig.suptitle(f"Workload {wid} - Difference of Polaris-cpu utilization MAE of GreedyMean & NNLS is largest", fontsize = 16)


    if len(nodes) == 1:
        axes = np.array([axes])  # keep 2D shape

    for i, node in enumerate(nodes):
        ax_tasks = axes[i, 0]
        ax_util = axes[i, 1]

        # ---------- left: task durations ----------
        intervals = normalized_task_durations.get(node, [])
        if intervals:
            task_ids = [t[0] for t in intervals]
            starts = np.array([t[1] for t in intervals], dtype=float)
            durs = np.array([t[2] for t in intervals], dtype=float)
            y = np.arange(len(intervals))

            ax_tasks.barh(y, durs, left=starts)
            ax_tasks.set_yticks(y)
            ax_tasks.set_yticklabels(task_ids)
        

            ax_tasks.set_xlabel("Time (relative)")
            ax_tasks.set_title(f"{node} — Task durations")

        # ---------- right: utilization ----------
        ts = np.asarray(node_series_norm[node]["timestamps"], dtype=float)
        util = np.asarray(node_series_norm[node]["util"], dtype=float)
        ax_util.plot(ts, util, label="Original")

        recon_nnls = recon_nnls_all.get(node, None)
        if recon_nnls is not None:
            ax_util.plot(ts, np.asarray(recon_nnls, dtype=float), label="Recon NNLS")

        recon_gm = recon_gm_all.get(node, None)
        if recon_gm is not None:
            ax_util.plot(ts, np.asarray(recon_gm, dtype=float), label="Recon GreedyMean")

        ax_util.set_xlabel("Time (relative)")
        ax_util.set_ylabel("Utilization")
        ax_util.set_title(f"{node} — Utilization")
        ax_util.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"b_cpu17max.png", dpi=200, bbox_inches="tight")
    plt.show()