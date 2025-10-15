#3 11 12 15 18 21 24

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA

# Open data files for reading
with open("all_workloads_ic2.json", "r") as f:
    workloads = json.load(f)

with open("all_system_loads_ic2.json", "r") as f:
    sysloads = json.load(f)

# choose workload by index
workload = workloads[17]
tasks = workload["tasklist"]

sys_entry = sysloads[17]
node_list = sys_entry["node_list"]

# detect the two node names in workload
node_names = []
unique_node = set()
for task in tasks: # in task list in workload file
    for node in task["nodes"]: # in node list
        n = node["node_name"] # Get node name
        # Get the name of the nodes in the workload
        # Since there are only two nodes, stop when have both
        if n not in unique_node: 
            unique_node.add(n)
            node_names.append(n)
        if len(node_names) == 2:
            break
    if len(node_names) == 2:
        break

# Break them
node1_name, node2_name = node_names[0], node_names[1]

# list of all start/end times for both nodes
start_end_node1 = []
start_end_node2 = []

# Get start/end times for both nodes
for task in tasks:
    s = float(task["start_time"])
    e = float(task["finish_time"])
    for node in task["nodes"]:
        n = node["node_name"]
        if n == node1_name:
            start_end_node1.extend([s, e])
        elif n == node2_name:
            start_end_node2.extend([s, e])

# Sort them
start_end_node1.sort()
start_end_node2.sort()

# deduplicate and sort times
times1 = sorted(set(start_end_node1))
times2 = sorted(set(start_end_node2))

# events: +1 at start, -1 at end
events_node1 = []
events_node2 = []

for task in tasks:
    s = float(task["start_time"])
    e = float(task["finish_time"])
    for node in task["nodes"]:
        n = node["node_name"]
        if n == node1_name:
            events_node1.append((s, +1)); events_node1.append((e, -1))
        elif n == node2_name:
            events_node2.append((s, +1)); events_node2.append((e, -1))
'''
 events_node1 = [
  (10, +1),  # Job A starts
  (20, -1),  # Job A ends
  (15, +1),  # Job B starts
  (25, -1),  # Job B ends
  (20, +1),  # Job C starts
  (30, -1)   # Job C ends]
'''
  
# sort events; process END before START at the same time
events_node1.sort(key=lambda x: (x[0], 0 if x[1] == -1 else 1))
events_node2.sort(key=lambda x: (x[0], 0 if x[1] == -1 else 1))
# Sort time first, then by type (-1 before +1), if happen at same time
# Subtract first because if not
# 10-20 20-30, 20s will have overlap

# no. of job running at time
counts_left1 = {}
counts_left2 = {}

active = 0; i = 0
# times1 is all unique start/end times for node1
for t in times1:
    # process all events, happens at this same time t, and 
    while i < len(events_node1) and events_node1[i][0] == t and events_node1[i][1] == -1:
        active += events_node1[i][1]; i += 1
    while i < len(events_node1) and events_node1[i][0] == t and events_node1[i][1] == +1:
        active += events_node1[i][1]; i += 1
    counts_left1[t] = active

'''
counts_left1 = {
  10: 1,  # 1 job active after 10
  15: 2,  # 2 jobs active after 15
  20: 1,  # 1 job active after 20
  25: 0   # 0 jobs active after 25
}
'''    

active = 0; i = 0
for t in times2:
    while i < len(events_node2) and events_node2[i][0] == t and events_node2[i][1] == -1:
        active += events_node2[i][1]; i += 1
    while i < len(events_node2) and events_node2[i][0] == t and events_node2[i][1] == +1:
        active += events_node2[i][1]; i += 1
    counts_left2[t] = active


    

# start-end-how_many_jobs
intervals_node1 = [(times1[i], times1[i+1], counts_left1[times1[i]]) for i in range(len(times1)-1)]
intervals_node2 = [(times2[i], times2[i+1], counts_left2[times2[i]]) for i in range(len(times2)-1)]

# pull CPU util series per node
def get_cpu_series_for_node(nm):
    for nd in node_list:
        if nd["node_name"] == nm:
            arr = nd["metrics"]["cpu_util"]  # list of [timestamp, util]
            t = np.array([float(x[0]) for x in arr])
            y = np.array([float(x[1]) for x in arr])
            return t, y
    return np.array([]), np.array([])

t1, y1 = get_cpu_series_for_node(node1_name)
t2, y2 = get_cpu_series_for_node(node2_name)

# FASTICA on intervals with overlap 2 or 3
# y is array of time, y is array of utilization, time window, overlap count
# --- Drop-in: replace your run_interval_ica with this ---
def run_interval_ica(t, y, tL, tR, k):
    # select the mixed segment in [tL, tR]
    mask = (t >= tL) & (t <= tR)
    idx = pd.Index(t[mask], name="time")
    mixed_segment = pd.Series(y[mask], index=idx)

    # 1-overlap: keep your original handling outside (nothing to separate here)
    if k == 1:
        # just return the mixed segment so you can plot/use your original single-signal line
        return {
            "interval": (tL, tR),
            "k": 1,
            "cpu_segment_time": idx,
            "cpu_segment": mixed_segment,
        }

    # 2-overlap: lag_0, lag_1 -> ICA(2)
    elif k == 2:
        # skip very short intervals
        if mixed_segment.size < (k + 5):
            return None

        mixed_matrix = pd.DataFrame({
            'lag_0': mixed_segment,
            'lag_1': mixed_segment.shift(1),
        }).dropna()

        mixed_matrix = mixed_matrix.replace([np.inf, -np.inf], np.nan).dropna()
        if len(mixed_matrix) < 2 or mixed_matrix.std().min() == 0:
            return None

        ica = FastICA(n_components=2, whiten='arbitrary-variance', random_state=42)
        S = ica.fit_transform(mixed_matrix.values)         # (T,2)
        A = ica.mixing_                                    # (2,2)
        mean0 = float(ica.mean_[0])                        # baseline of lag_0

        # project sources onto lag_0 channel
        c0 = S[:, 0] * A[0, 0]                             # (T,)
        c1 = S[:, 1] * A[0, 1]                             # (T,)
        contribs = np.column_stack([c0, c1])               # (T,2)

        # enforce exact add-back to the observed lag_0
        y_obs = mixed_matrix['lag_0'].values               # (T,)
        eps = 1e-12
        mag = np.abs(contribs)                   # nonnegative proxy for each component's share
        row_sum = mag.sum(axis=1, keepdims=True)
        row_sum[row_sum < eps] = 1.0             # avoid /0: if both ~0, split evenly below
        weights = mag / row_sum                  # each row sums to 1

        # where both mags ~0, make it exactly 50/50 to be stable
        both_zero = (mag[:, 0] < eps) & (mag[:, 1] < eps)
        if both_zero.any():
            weights[both_zero, 0] = 0.5
            weights[both_zero, 1] = 0.5

        parts = weights * y_obs[:, None]       

        S1_separated = pd.Series(parts[:, 0], index=mixed_matrix.index)
        S2_separated = pd.Series(parts[:, 1], index=mixed_matrix.index)

        return {
            "interval": (tL, tR),
            "k": 2,
            "cpu_segment_time": mixed_matrix.index,
            "cpu_segment": mixed_matrix['lag_0'],
            "S1": S1_separated,
            "S2": S2_separated,
            "components_sum": S1_separated + S2_separated
        }
    # 3-overlap: lag_0, lag_1, lag_2 -> ICA(3)
    elif k == 3:
        if mixed_segment.size < (k + 5):
            return None

        mixed_matrix = pd.DataFrame({
            'lag_0': mixed_segment,
            'lag_1': mixed_segment.shift(1),
            'lag_2': mixed_segment.shift(2),
        }).dropna()

        mixed_matrix = mixed_matrix.replace([np.inf, -np.inf], np.nan).dropna()
        if len(mixed_matrix) < 3 or mixed_matrix.std().min() == 0:
            return None

        ica = FastICA(n_components=3, whiten='arbitrary-variance', random_state=42)
        S = ica.fit_transform(mixed_matrix.values)         # (T,3)
        A = ica.mixing_                                    # (3,3)
        mean0 = float(ica.mean_[0])

        # project onto lag_0
        c0 = S[:, 0] * A[0, 0]
        c1 = S[:, 1] * A[0, 1]
        c2 = S[:, 2] * A[0, 2]
        contribs = np.column_stack([c0, c1, c2])           # (T,3)

        y_obs = mixed_matrix['lag_0'].values
        eps = 1e-12
        mag = np.abs(contribs)
        row_sum = mag.sum(axis=1, keepdims=True)
        row_sum[row_sum < eps] = 1.0
        weights = mag / row_sum

        # if all three are ~0 at some t, split 1/3 each
        all_zero = (mag[:, 0] < eps) & (mag[:, 1] < eps) & (mag[:, 2] < eps)
        if all_zero.any():
            weights[all_zero, :] = 1.0 / 3.0

        parts = weights * y_obs[:, None]           # (T,3)

        S1_separated = pd.Series(parts[:, 0], index=mixed_matrix.index)
        S2_separated = pd.Series(parts[:, 1], index=mixed_matrix.index)
        S3_separated = pd.Series(parts[:, 2], index=mixed_matrix.index)

        return {
            "interval": (tL, tR),
            "k": 3,
            "cpu_segment_time": mixed_matrix.index,
            "cpu_segment": mixed_matrix['lag_0'],
            "S1": S1_separated,
            "S2": S2_separated,
            "S3": S3_separated,
            "components_sum": S1_separated + S2_separated + S3_separated
        }

    # k >= 4 not handled here (your file focuses on 1/2/3)
    return None
results_node1 = []
results_node2 = []

for (tL, tR, k) in intervals_node1:
    if k in (2, 3):
        r = run_interval_ica(t1, y1, tL, tR, k)
        if r is not None:
            results_node1.append(r)

for (tL, tR, k) in intervals_node2:
    if k in (2, 3):
        r = run_interval_ica(t2, y2, tL, tR, k)
        if r is not None:
            results_node2.append(r)

# After this:
# - intervals_node1 / intervals_node2 hold (t_left, t_right, overlap_count)
# - results_node1 / results_node2 are lists of dicts with:
#   interval, k, cpu_segment_time, cpu_segment, components (k arrays), components_sum, reconstruction, mse_vs_cpu, mae_vs_cpuu

# ------------------ Plot both nodes with overlaps and ICA ------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("CPU Utilization and ICA-separated components per Node", fontsize=14)

# ===== Node 1 =====
ax = axes[0]
ax.plot(t1, y1, color='black', linewidth=1.2, label=f'{node1_name} CPU Utilization')

# shade intervals with 2/3 overlaps
for (tL, tR, k) in intervals_node1:
    if k in (2, 3):
        color = 'red' if k == 2 else 'orange'
        ax.axvspan(tL, tR, color=color, alpha=0.25)

# plot separated components returned by run_interval_ica
for r in results_node1:
    mu = float(np.mean(r["cpu_segment"]))  # optional baseline so components sit near the mixed curve
    if "S1" in r:
        ax.plot(r["S1"].index, r["S1"].values + mu, linestyle='--', linewidth=1.0, label="S1 (node1)")
    if "S2" in r:
        ax.plot(r["S2"].index, r["S2"].values + mu, linestyle='--', linewidth=1.0, label="S2 (node1)")
    if "S3" in r:
        ax.plot(r["S3"].index, r["S3"].values + mu, linestyle='--', linewidth=1.0, label="S3 (node1)")

ax.set_ylabel('CPU Utilization')
ax.set_title(f'Node: {node1_name}')
ax.legend(loc='upper right')

# ===== Node 2 =====
ax = axes[1]
ax.plot(t2, y2, color='black', linewidth=1.2, label=f'{node2_name} CPU Utilization')

for (tL, tR, k) in intervals_node2:
    if k in (2, 3):
        color = 'red' if k == 2 else 'orange'
        ax.axvspan(tL, tR, color=color, alpha=0.25)

for r in results_node2:
    mu = float(np.mean(r["cpu_segment"]))
    if "S1" in r:
        ax.plot(r["S1"].index, r["S1"].values + mu, linestyle='--', linewidth=1.0, label="S1 (node2)")
    if "S2" in r:
        ax.plot(r["S2"].index, r["S2"].values + mu, linestyle='--', linewidth=1.0, label="S2 (node2)")
    if "S3" in r:
        ax.plot(r["S3"].index, r["S3"].values + mu, linestyle='--', linewidth=1.0, label="S3 (node2)")

ax.set_xlabel('Time')
ax.set_ylabel('CPU Utilization')
ax.set_title(f'Node: {node2_name}')
ax.legend(loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
plt.savefig("zz17.png", dpi=300)