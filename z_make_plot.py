import json
import matplotlib.pyplot as plt

with open("all_workloads_ic2.json", 'r') as f:
    workloads= json.load(f)

for i in range(len(workloads)):
    tasks = workloads[i]["tasklist"]

    node_to_intervals = {}

    for task in tasks:
        start_time = float(task["start_time"])
        finish_time = float(task["finish_time"])
        task_id = task["task_id"]
        duration = finish_time - start_time

        for node in task.get("nodes", []):
            node_name = node["node_name"]
            node_to_intervals.setdefault(node_name, []).append((task_id, start_time, duration))


    n_nodes = len(node_to_intervals)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(12, 3 * n_nodes), sharex=True)

    if n_nodes == 1:
        axes = [axes]

    for ax, (node_name, intervals) in zip(axes, node_to_intervals.items()):
        task_ids, starts, durations = zip(*intervals)
        ax.barh(task_ids, durations, left=starts, height = 0.6, color='skyblue')
        ax.set_title(f"Task on {node_name}")
        ax.set_ylabel("Task ID")
        ax.invert_yaxis()

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"data_workload{i}.png", dpi=300)