import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA

# load data files
with open("all_workloads_ic2.json", "r") as f:
    workloads = json.load(f)

with open("all_system_loads_ic2.json", "r") as f:
    sysloads = json.load(f)

# ------------------ choose workload by index ------------------
WIDX = 11
workload = workloads[WIDX]
tasks = workload["tasklist"]
node_list = sysloads[WIDX]["node_list"]

# ------------------ detect the two node names in this workload ------------------
node_names = []
seen = set()
for task in tasks:
    for node in task["nodes"]:
        n = node["node_name"]
        if n not in seen:
            seen.add(n)
            node_names.append(n)
        if len(node_names) == 2:
            break
    if len(node_names) == 2:
        break

node1_name, node2_name = node_names[0], node_names[1]

# ------------------ collect all start/end times per node ------------------
start_end_node1, start_end_node2 = [], []
for task in tasks:
    s = float(task["start_time"])
    e = float(task["finish_time"])
    for node in task["nodes"]:
        n = node["node_name"]
        if n == node1_name:
            start_end_node1.extend([s, e])
        elif n == node2_name:
            start_end_node2.extend([s, e])

start_end_node1.sort()
start_end_node2.sort()
times1 = sorted(set(start_end_node1))
times2 = sorted(set(start_end_node2))

# ------------------ build start/end events and overlap counts ------------------
events_node1, events_node2 = [], []
for task in tasks:
    s = float(task["start_time"])
    e = float(task["finish_time"])
    for node in task["nodes"]:
        n = node["node_name"]
        if n == node1_name:
            events_node1.append((s, +1)); events_node1.append((e, -1))
        elif n == node2_name:
            events_node2.append((s, +1)); events_node2.append((e, -1))

# sort by time; at equal time process END (-1) before START (+1)
events_node1.sort(key=lambda x: (x[0], 0 if x[1] == -1 else 1))
events_node2.sort(key=lambda x: (x[0], 0 if x[1] == -1 else 1))

counts_left1, counts_left2 = {}, {}

active, i = 0, 0
for t in times1:
    while i < len(events_node1) and events_node1[i][0] == t and events_node1[i][1] == -1:
        active += events_node1[i][1]; i += 1
    while i < len(events_node1) and events_node1[i][0] == t and events_node1[i][1] == +1:
        active += events_node1[i][1]; i += 1
    counts_left1[t] = active

active, i = 0, 0
for t in times2:
    while i < len(events_node2) and events_node2[i][0] == t and events_node2[i][1] == -1:
        active += events_node2[i][1]; i += 1
    while i < len(events_node2) and events_node2[i][0] == t and events_node2[i][1] == +1:
        active += events_node2[i][1]; i += 1
    counts_left2[t] = active

intervals_node1 = [(times1[i], times1[i+1], counts_left1[times1[i]]) for i in range(len(times1)-1)]
intervals_node2 = [(times2[i], times2[i+1], counts_left2[times2[i]]) for i in range(len(times2)-1)]

# ------------------ pull CPU util series per node ------------------
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

# ------------------ preallocate time-indexed Series for separated sources ------------------
# (NaNs by default; we will fill only the times inside overlap intervals)
s1_node1 = pd.Series(index=t1, dtype=float)
s2_node1 = pd.Series(index=t1, dtype=float)
s3_node1 = pd.Series(index=t1, dtype=float)  # used only if k == 3

s1_node2 = pd.Series(index=t2, dtype=float)
s2_node2 = pd.Series(index=t2, dtype=float)
s3_node2 = pd.Series(index=t2, dtype=float)  # used only if k == 3

# ------------------ run FastICA per overlap interval and assign with .loc ------------------
# minimal guards to avoid numerical blow-ups
MIN_SAMPLES = 10  # raise if needed for stability

# Node 1
for (tL, tR, k) in intervals_node1:
    if k in (2, 3):
        mask = (t1 >= tL) & (t1 <= tR)
        tseg = t1[mask]
        seg = y1[mask]
        if seg.size < max(MIN_SAMPLES, k + 5):
            continue

        seg_c = seg - np.mean(seg)

        # Build lagged DataFrame with time index so .loc works on timestamps
        mixed_matrix = pd.DataFrame({"lag_0": seg_c}, index=tseg)
        for d in range(1, k):
            mixed_matrix[f"lag_{d}"] = mixed_matrix["lag_0"].shift(d)

        mixed_matrix = mixed_matrix.dropna()
        if mixed_matrix.empty:
            continue

        X = mixed_matrix.values
        # simple variance guard to avoid singular whitening
        if np.any(np.std(X, axis=0) < 1e-10):
            continue

        ica = FastICA(n_components=k, whiten='unit-variance', random_state=0, max_iter=1000)
        separated_segment = ica.fit_transform(X)  # shape: (M, k)

        # assign back by time index
        s1_node1.loc[mixed_matrix.index] = separated_segment[:, 0]
        s2_node1.loc[mixed_matrix.index] = separated_segment[:, 1]
        if k == 3:
            s3_node1.loc[mixed_matrix.index] = separated_segment[:, 2]

# Node 2
for (tL, tR, k) in intervals_node2:
    if k in (2, 3):
        mask = (t2 >= tL) & (t2 <= tR)
        tseg = t2[mask]
        seg = y2[mask]
        if seg.size < max(MIN_SAMPLES, k + 5):
            continue

        seg_c = seg - np.mean(seg)

        mixed_matrix = pd.DataFrame({"lag_0": seg_c}, index=tseg)
        for d in range(1, k):
            mixed_matrix[f"lag_{d}"] = mixed_matrix["lag_0"].shift(d)

        mixed_matrix = mixed_matrix.dropna()
        if mixed_matrix.empty:
            continue

        X = mixed_matrix.values
        if np.any(np.std(X, axis=0) < 1e-10):
            continue

        ica = FastICA(n_components=k, whiten='unit-variance', random_state=0, max_iter=1000)
        separated_segment = ica.fit_transform(X)

        s1_node2.loc[mixed_matrix.index] = separated_segment[:, 0]
        s2_node2.loc[mixed_matrix.index] = separated_segment[:, 1]
        if k == 3:
            s3_node2.loc[mixed_matrix.index] = separated_segment[:, 2]

# ------------------ plot (no overlap shading) ------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("CPU Utilization and Separated Sources (FastICA)")

# Node 1
ax = axes[0]
ax.plot(t1, y1, color='black', linewidth=1.2, label=f'{node1_name} CPU')
ax.plot(s1_node1.index, s1_node1.values, linewidth=1.0, label='S1')
ax.plot(s2_node1.index, s2_node1.values, linewidth=1.0, label='S2')
if not s3_node1.isna().all():
    ax.plot(s3_node1.index, s3_node1.values, linewidth=1.0, label='S3')
ax.set_ylabel('Value')
ax.set_title(f'Node: {node1_name}')
ax.legend(loc='upper right')

# Node 2
ax = axes[1]
ax.plot(t2, y2, color='black', linewidth=1.2, label=f'{node2_name} CPU')
ax.plot(s1_node2.index, s1_node2.values, linewidth=1.0, label='S1')
ax.plot(s2_node2.index, s2_node2.values, linewidth=1.0, label='S2')
if not s3_node2.isna().all():
    ax.plot(s3_node2.index, s3_node2.values, linewidth=1.0, label='S3')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title(f'Node: {node2_name}')
ax.legend(loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("t11.png", dpi=300)
plt.show()