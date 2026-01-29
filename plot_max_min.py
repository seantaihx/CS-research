import json
import matplotlib.pyplot as plt
from algorithm_blockwise_nonnegative_least_squares import mean_then_blockwise_nnls
from algorithm_greedy_mean_only import greedy_mean_separation
import numpy as np
from algorithm_blockwise_nonnegative_least_squares import nnls_main
from algorithm_greedy_mean_only import gm_main


def task_durations_data(workloads, wid):

    tasks = workloads[wid]["tasklist"]
    # build node -> [(task_id, start, duration), ...]
    node_to_intervals = {}
    node_min_start = {}

    for task in tasks:
        start_time = float(task["start_time"])
        finish_time = float(task["finish_time"])
        task_id = task["task_id"]
        duration = finish_time - start_time
        for node in task.get("nodes", []):
            node_name = node["node_name"]
            if node_name not in node_min_start:
                node_min_start[node_name] = start_time
            else:
                node_min_start[node_name] = min(node_min_start[node_name], start_time)
            node_to_intervals.setdefault(node_name, []).append((task_id, start_time, duration))

    normalized_timestamps = {}
    for node, intervals in node_to_intervals.items():
        normalized_timestamps[node] = []
        min_start = node_min_start[node]
        for task, start, duration in intervals:
            normalized_start = start - min_start
            normalized_timestamps[node].append((task, normalized_start, duration))

    #print(normalized_timestamps)
    return normalized_timestamps

def max_min(workload_file, system_file, util):
    nnls_mae_list = nnls_main(workload_file, system_file, util)
    gm_mae_list = gm_main(workload_file, system_file, util)
    difference_list = []
    max_index = 0
    min_index = 0
    max_value = 0.0
    min_value = 0.0
    difference = 0.0
    for i in range(len(nnls_mae_list)):
        
        difference_list.append(abs(nnls_mae_list[i] - gm_mae_list[i]))
        if difference_list[i] > max_value or max_value == 0.0:
            max_index = i
            max_value = difference_list[i]
        if difference_list[i] < min_value or min_value == 0.0:
            min_index = i
            min_value = difference_list[i]
        

    
    #print(nnls_mae_list)
    #print()
    #print(gm_mae_list)
    '''
    print()
    print(difference_list)
    print()
    print(max_index, min_index)
    print()
    print(max_value, min_value)
    '''
    return max_index, min_index




def all_util_data(workloads, system_loads, wid, which_util):
    

    workload = workloads[wid]
    system_load = system_loads[wid]
    node_series = {}
    t, v, recon_nnls, recon_gm = None, None, None, None
    for node in system_load.get("node_list", []):
        name = node.get("node_name")
        pairs = node.get("metrics", {}).get(f"{which_util}_util", [])
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

    tasks = [{
        "task_id": int(t["task_id"]),
        "start": float(t["start_time"]),
        "finish": float(t["finish_time"]),
        "nodes": [n["node_name"] for n in t.get("nodes", []) if n.get("node_name")]
    } for t in workload.get("tasklist", [])]

    recon_gm_all = {}
    recon_nnls_all = {}
    for node, data in node_series.items():
        node_task = [t for t in tasks if node in t["nodes"]]
        if not node_task:
            continue
        contribs_nnls = mean_then_blockwise_nnls(
            data["timestamps"], data["util"], node_task, 30.0, 1e-2)
        #print(contribs_nnls)
        recon_gm = greedy_mean_separation(
            data["timestamps"], data["util"], node_task)
        recon_nnls = sum(contribs_nnls.values())
        recon_gm_all[node] = recon_gm
        recon_nnls_all[node] = recon_nnls
        #print(recon_nnls)
        #print(recon_gm)
    node_series_norm = {}
    for name, data in node_series.items():
        ts = data["timestamps"]
        t0 = ts.min()

        node_series_norm[name] = {
            "timestamps": ts - t0,
            "util": data["util"]
        }

        #print(recon_nnls_all)

    #for name, data in node_series_norm.items():
        #print(data)

    return node_series_norm, recon_nnls_all, recon_gm_all

def plot_polaris(workload_file, system_file, wid, util, word, num_nodes):
    normalized_task_durations = task_durations_data(workload_file, wid)
    node_series_norm, recon_nnls_all, recon_gm_all = all_util_data(workload_file, system_file, wid, util)
    nodes = list(node_series_norm.keys())[:num_nodes]
    fig, axes = plt.subplots(len(nodes), 2, figsize=(18, 2.2 * len(nodes)), sharex = False)
    if util == 'cpu':
        if word == "max":
            plt.suptitle(f"Workload {wid} - Difference of Polaris-cpu utilization MAE of GreedyMean & NNLS is largest", fontsize = 16)
        elif word == "min":
            plt.suptitle(f"Workload {wid} - Difference of Polaris-cpu utilization MAE of GreedyMean & NNLS is smallest", fontsize = 16)
    elif util == 'memory':
        if word == "max":
            plt.suptitle(f"Workload {wid} - Difference of Polaris-memory utilization MAE of GreedyMean & NNLS is largest", fontsize = 16)
        elif word == "min":
            plt.suptitle(f"Workload {wid} - Difference of Polaris-memory utilization MAE of GreedyMean & NNLS is smallest", fontsize = 16)

    if len(nodes) == 1:
        axes = np.array([axes])
    
    for i, node in enumerate(nodes):
        ax_tasks = axes[i, 0]
        ax_util = axes[i, 1]

        intervals = normalized_task_durations.get(node, [])
        if intervals:
            task_ids = [t[0] for t in intervals]
            starts = np.array([t[1] for t in intervals], dtype=float)
            durations = np.array([t[2] for t in intervals], dtype=float)

            y = np.arange(len(intervals))

            ax_tasks.barh(y, durations, left=starts)
            ax_tasks.set_yticks(y)
            ax_tasks.set_yticklabels(task_ids)
            ax_tasks.set_xlabel("Time")
            ax_tasks.set_title(f"Task Durations on {node}")

        ts = np.asarray(node_series_norm[node]["timestamps"], dtype=float)
        vals = np.asarray(node_series_norm[node]["util"], dtype=float)
        ax_util.plot(ts, vals, label="Observed Util")
        recon_nnls = recon_nnls_all.get(node, None)
        if recon_nnls is not None:
            ax_util.plot(ts, np.asarray(recon_nnls, dtype=float), label="NNLS Recon", linestyle="--")

        recon_gm = recon_gm_all.get(node, None)
        if recon_gm is not None:
            ax_util.plot(ts, np.asarray(recon_gm, dtype=float), label="Greedy Mean Recon", linestyle=":")

        if util == 'cpu':
            ax_util.set_ylim(0, 100)
        elif util == 'memory':
            ax_util.set_ylim(0, 50)
        ax_util.set_xlabel("Time")
        ax_util.set_ylabel("Utilization")
        ax_util.set_title(f"Utilization on {node}")
        ax_util.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"polaris-{util}_{word}_{wid}.png")

def plot_ic2(workload_file, system_file, wid, util, word):
    #workload_file = "all_workloads_ic2.json"
    #system_file = "all_system_loads_ic2.json"
    #workload_file = input("Enter workload file: ")
    #system_file = input("Enter system load file: ")
    #wid = 15
    #max_index, min_index = max_min(workload_file, system_file, util)
    normalized_task_durations = task_durations_data(workload_file, wid)
    node_series_norm, recon_nnls_all, recon_gm_all = all_util_data(workload_file, system_file, wid, util)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if util == 'cpu':
        if word == "max":
            fig.suptitle(f"Workload {wid} - Difference of IC2-cpu utilization MAE of GreedyMean & NNLS is largest", fontsize = 16)
        elif word == "min":
            fig.suptitle(f"Workload {wid} - Difference of IC2-cpu utilization MAE of GreedyMean & NNLS is smallest", fontsize = 16)
    elif util == 'memory':
        if word == "max":
            fig.suptitle(f"Workload {wid} - Difference of IC2-memory utilization MAE of GreedyMean & NNLS is largest", fontsize = 16)
        elif word == "min":
            fig.suptitle(f"Workload {wid} - Difference of IC2-memory utilization MAE of GreedyMean & NNLS is smallest", fontsize = 16)

    nodes = list(node_series_norm.keys())
    for i, node in enumerate(nodes):
        ax_tasks = axes[0, i]
        ax_util = axes[1, i]

        intervals = normalized_task_durations.get(node, [])
        if intervals:
            task_ids = [t[0] for t in intervals]
            starts = np.array([t[1] for t in intervals], dtype=float)
            durations = np.array([t[2] for t in intervals], dtype=float)

            y = np.arange(len(intervals))

            ax_tasks.barh(y, durations, left=starts)
            ax_tasks.set_yticks(y)
            ax_tasks.set_yticklabels(task_ids)
            ax_tasks.set_xlabel("Time")
            ax_tasks.set_title(f"Task Durations on {node}")

        ts = np.asarray(node_series_norm[node]["timestamps"], dtype=float)
        vals = np.asarray(node_series_norm[node]["util"], dtype=float)
        ax_util.plot(ts, vals, label="Original Util", color="black", linewidth=1.5)
        
        recon_nnls = recon_nnls_all.get(node, None)
        if recon_nnls is not None:
            ax_util.plot(ts, np.asarray(recon_nnls, dtype=float), label="NNLS Recon", linestyle="--")

        recon_gm = recon_gm_all.get(node, None)
        if recon_gm is not None:
            ax_util.plot(ts, np.asarray(recon_gm, dtype=float), label="Greedy Mean Recon", linestyle=":")


        if util == 'cpu':
            ax_util.set_ylim(0, 55)
        elif util == 'memory':
            ax_util.set_ylim(0, 10)
        
        ax_util.set_xlabel("Time")
        ax_util.set_ylabel("Utilization")
        ax_util.set_title(f"Utilization on {node}")
        ax_util.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"ic2-{util}_{word}_{wid}.png")

if __name__ == "__main__":
    
    prompt = input("What to plot? (ic2/polaris/both): ").strip().lower()
    if prompt == "ic2":
        workload_file = input("Enter IC2 workload file: ")
        system_file = input("Enter IC2 system load file: ")
        with open(workload_file, "r") as f1:
            workloads = json.load(f1)
        with open(system_file, "r") as f2:
            system_loads = json.load(f2)
        max_index_cpu, min_index_cpu = max_min(workloads, system_loads, "cpu")
        max_index_mem, min_index_mem = max_min(workloads, system_loads, "memory")
        
        plot_ic2(workloads, system_loads, min_index_cpu, util = 'cpu', word = "min")
        plot_ic2(workloads, system_loads, max_index_cpu, util = 'cpu', word = "max")
        plot_ic2(workloads, system_loads, max_index_mem, util = 'memory', word = "max")
        plot_ic2(workloads, system_loads, min_index_mem, util = 'memory', word = "min")
    elif prompt == "polaris":
        workload_file = input("Enter Polaris workload file: ")
        system_file = input("Enter Polaris system load file: ")
        with open(workload_file, "r") as f1:
            workloads = json.load(f1)
        with open(system_file, "r") as f2:
            system_loads = json.load(f2)
        max_index_cpu, min_index_cpu = max_min(workloads, system_loads, "cpu")
        max_index_mem, min_index_mem = max_min(workloads, system_loads, "memory")

        num_nodes = int(input("Enter number of nodes to plot: "))
        plot_polaris(workloads, system_loads, min_index_cpu, util = 'cpu', word = "min", num_nodes = num_nodes)
        plot_polaris(workloads, system_loads, max_index_cpu, util = 'cpu', word = "max", num_nodes = num_nodes)
        plot_polaris(workloads, system_loads, max_index_mem, util = 'memory', word = "max", num_nodes = num_nodes)
        plot_polaris(workloads, system_loads, min_index_mem, util = 'memory', word = "min", num_nodes = num_nodes)
    elif prompt == "both":
        workload_file_ic2 = input("Enter IC2 workload file: ")
        system_file_ic2 = input("Enter IC2 system load file: ")
        workload_file_polaris = input("Enter Polaris workload file: ")
        system_file_polaris = input("Enter Polaris system load file: ")
        with open(workload_file_ic2, "r") as f1:
            workloads_ic2 = json.load(f1)
        with open(system_file_ic2, "r") as f2:
            system_loads_ic2 = json.load(f2)
        with open(workload_file_polaris, "r") as f3:
            workloads_polaris = json.load(f3)
        with open(system_file_polaris, "r") as f4:
            system_loads_polaris = json.load(f4)
        max_index_ic2_cpu, min_index_ic2_cpu = max_min(workloads_ic2, system_loads_ic2, "cpu")
        max_index_ic2_mem, min_index_ic2_mem = max_min(workloads_ic2, system_loads_ic2, "memory")
        max_index_polaris_cpu, min_index_polaris_cpu = max_min(workloads_polaris, system_loads_polaris, "cpu")
        max_index_polaris_mem, min_index_polaris_mem = max_min(workloads_polaris, system_loads_polaris, "memory")
        num_nodes = int(input("Enter number of nodes to plot for Polaris: "))
        plot_ic2(workloads_ic2, system_loads_ic2, min_index_ic2_cpu, util = 'cpu', word = "min")
        plot_ic2(workloads_ic2, system_loads_ic2, max_index_ic2_cpu, util = 'cpu', word = "max")
        plot_ic2(workloads_ic2, system_loads_ic2, max_index_ic2_mem, util = 'memory', word = "max")
        plot_ic2(workloads_ic2, system_loads_ic2, min_index_ic2_mem, util = 'memory', word = "min")
        plot_polaris(workloads_polaris, system_loads_polaris, min_index_polaris_cpu, util = 'cpu', word = "min", num_nodes = num_nodes)
        plot_polaris(workloads_polaris, system_loads_polaris, max_index_polaris_cpu, util = 'cpu', word = "max", num_nodes = num_nodes)
        plot_polaris(workloads_polaris, system_loads_polaris, max_index_polaris_mem, util = 'memory', word = "max", num_nodes = num_nodes)
        plot_polaris(workloads_polaris, system_loads_polaris, min_index_polaris_mem, util = 'memory', word = "min", num_nodes = num_nodes)
        
    
    
