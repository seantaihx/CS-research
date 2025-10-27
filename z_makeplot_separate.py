import json
import matplotlib.pyplot as plt

# --- choose workload index manually ---
i = 1  # change this to the workload index you want

with open("all_workloads_ic2.json", 'r') as f:
    workloads = json.load(f)

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

# --- create one plot per node ---
for node_name, intervals in node_to_intervals.items():
    task_ids, starts, durations = zip(*intervals)
    plt.figure(figsize=(12, 3))
    plt.barh(task_ids, durations, left=starts, height=0.6, color='skyblue')
    plt.title(f"Tasks on {node_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Task ID")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"data_workload{i}_{node_name}.png", dpi=300)
    plt.close()