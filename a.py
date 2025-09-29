import matplotlib.pyplot as plt
import json
import numpy as np
path = "all_workloads_ic2.json"

def visualize_runtime_cpus_gpus(path):
    start_time = []
    finish_time = [] 
    runtimes = []
    task_id = []
    node_names = []
    with open(path, 'r') as f:
        workloads= json.load(f)
    
    workload = workloads[0]

    tasklist = workload["tasklist"]
    #count = 0

    for task in tasklist:
        nodelist = task["nodes"]
        start_time.append(float(task["start_time"]))
        finish_time.append(float(task["finish_time"]))
        task_id.append(task["task_id"])
        for node in nodelist:
            nodename = node["node_name"]
            node_names.append(nodename)
            rt = (float(task["finish_time"]) - float(task["start_time"]))
            runtimes.append(rt)

    min_start = min(start_time)
    start_time = [i - min_start for i in start_time]
    finish_time = [i - min_start for i in finish_time] 

    node_to_bars = {}
    for s, r, n in zip(start_time, runtimes, node_names):
        if n not in node_to_bars:
            node_to_bars[n] = []
        node_to_bars[n].append((s, r))

    # Assign each node a row index
    y_index = {node: i for i, node in enumerate(sorted(node_to_bars.keys()))}

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(y_index)))

    # Draw all bars
    for node, bars in node_to_bars.items():
        row = y_index[node]
        ax.broken_barh(bars, (row - 0.4, 0.8), facecolors="skyblue")

    # Y-axis with node names
    ax.set_yticks(range(len(y_index)))
    ax.set_yticklabels(sorted(y_index.keys()))

    ax.set_xlabel("Time since earliest start (s)")
    ax.set_ylabel("Node name")
    ax.set_title("Task runtime by node")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
    plt.savefig("gantt_chart.png", dpi=300)
    print(task_id)
    return node_names


'''

    y_axis_runtime = [small_runtimes, scattered_runtimes, mixed_runtimes, scalable_runtimes, threaded_runtimes]
    axs[0].set_yscale('log')
    axs[0].boxplot(y_axis_runtime, labels = x_axis)
    axs[0].set_xlabel("Workload Type")
    axs[0].set_ylabel("Minutes")
    axs[0].set_xticks(X_axis_len)
    axs[0].set_xticklabels(x_axis, fontsize=10, rotation=30)
    axs[0].set_title(cluster_name+" Cluster - Runtime vs. Workload Type",fontsize=10)
    
    y_axis_cpus = [small_cpus, scattered_cpus, mixed_cpus, scalable_cpus, threaded_cpus]

    return axs
    #plt.show()
'''
if __name__ == "__main__":

    n = visualize_runtime_cpus_gpus("all_workloads_ic2.json")
    print(n)

    


'''
    x_axis = ["small", "scattered", "mixed", "scalable", "threaded"]
    X_axis_len = np.arange(len(x_axis))
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    #fig.suptitle("", fontsize=12)
    #path_list = ["../data/all_workloads_ic2.json","../data/all_workloads_polaris.json","../data/all_workloads_aws.json"]
    cluster_name = "IC2"
    axs = visualize_runtime_cpus_gpus(path, cluster_name, axs)
    fig.tight_layout()
    plt.show()
    #plt.savefig("workload_stats.png", dpi=300)
'''

