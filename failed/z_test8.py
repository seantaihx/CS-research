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
workload = workloads[11]
tasks = workload["tasklist"]

sys_entry = sysloads[11]
node_list = sys_entry["node_list"]

# detect the two node names in workload
node_names = []
unique_node = set()
for task in tasks:
    for node in task["nodes"]:
        n = node["node_name"]
        if n not in unique_node:
            unique_node.add(n)
            node_names.append(n)
        if len(node_names) == 2:
            break
    if len(node_names) == 2:
        break

node1_name, node2_name = node_names[0], node_names[1]

# collect start/end times per node
start_end_node1, start_end_node2 = [], []
for task in tasks:
    s = float(task["start_time"]); e = float(task["finish_time"])
    for node in task["nodes"]:
        n = node["node_name"]
        if n == node1_name:
            start_end_node1.extend([s, e])
        elif n == node2_name:
            start_end_node2.extend([s, e])

start_end_node1.sort(); start_end_node2.sort()
times1 = sorted(set(start_end_node1))
times2 = sorted(set(start_end_node2))

# events
events_node1, events_node2 = [], []
for task in tasks:
    s = float(task["start_time"]); e = float(task["finish_time"])
    for node in task["nodes"]:
        n = node["node_name"]
        if n == node1_name:
            events_node1.append((s, +1)); events_node1.append((e, -1))
        elif n == node2_name:
            events_node2.append((s, +1)); events_node2.append((e, -1))

events_node1.sort(key=lambda x: (x[0], 0 if x[1] == -1 else 1))
events_node2.sort(key=lambda x: (x[0], 0 if x[1] == -1 else 1))

counts_left1, counts_left2 = {}, {}
active = 0; i = 0
for t in times1:
    while i < len(events_node1) and events_node1[i][0] == t and events_node1[i][1] == -1:
        active += events_node1[i][1]; i += 1
    while i < len(events_node1) and events_node1[i][0] == t and events_node1[i][1] == +1:
        active += events_node1[i][1]; i += 1
    counts_left1[t] = active

active = 0; i = 0
for t in times2:
    while i < len(events_node2) and events_node2[i][0] == t and events_node2[i][1] == -1:
        active += events_node2[i][1]; i += 1
    while i < len(events_node2) and events_node2[i][0] == t and events_node2[i][1] == +1:
        active += events_node2[i][1]; i += 1
    counts_left2[t] = active

intervals_node1 = [(times1[i], times1[i+1], counts_left1[times1[i]]) for i in range(len(times1)-1)]
intervals_node2 = [(times2[i], times2[i+1], counts_left2[times2[i]]) for i in range(len(times2)-1)]

# pull CPU util series per node
for nd in node_list:
    if nd["node_name"] == node1_name:
        arr = nd["metrics"]["cpu_util"]
        t_n1 = np.array([float(x[0]) for x in arr])
        u_n1 = np.array([float(x[1]) for x in arr])
    elif nd["node_name"] == node2_name:
        arr = nd["metrics"]["cpu_util"]
        t_n2 = np.array([float(x[0]) for x in arr])
        u_n2 = np.array([float(x[1]) for x in arr])

# Build time-indexed series
cpu1 = pd.Series(u_n1, index=pd.to_datetime(t_n1, unit='s'), name='cpu')
cpu2 = pd.Series(u_n2, index=pd.to_datetime(t_n2, unit='s'), name='cpu')

# ---- global means per node (THIS is the change) ----
mu1 = float(cpu1.mean())
mu2 = float(cpu2.mean())

# containers
base_n1 = pd.Series(0.0, index=cpu1.index)
base_n2 = pd.Series(0.0, index=cpu2.index)

sep1_n1 = pd.Series(0.0, index=cpu1.index)
sep2_n1 = pd.Series(0.0, index=cpu1.index)
sep3_n1 = pd.Series(0.0, index=cpu1.index)

sep1_n2 = pd.Series(0.0, index=cpu2.index)
sep2_n2 = pd.Series(0.0, index=cpu2.index)
sep3_n2 = pd.Series(0.0, index=cpu2.index)

# ============================
# Node 1
# ============================
for tL, tR, k in intervals_node1:
    if k == 2:
        m = (cpu1.index >= pd.to_datetime(tL, unit='s')) & (cpu1.index < pd.to_datetime(tR, unit='s'))
        s = cpu1.loc[m]
        if s.empty:
            continue

        mixed_matrix = pd.DataFrame({
            'lag_0': s - mu1,
            'lag_1': s.shift(1) - mu1,
        }).dropna()

        if len(mixed_matrix) < 3+5:
            continue

        x = mixed_matrix.values
        rank = np.linalg.matrix_rank(x)
        n_comp = min(2, mixed_matrix.shape[1], rank)
        if n_comp < 1:
            continue

        ica = FastICA(n_components=n_comp, whiten='arbitrary-variance', random_state=42)
        sIC = ica.fit_transform(x)
        A = ica.mixing_

        # ...
        c0 = pd.Series(sIC[:, 0] * A[0, 0], index=mixed_matrix.index)
        if n_comp > 1:
            c1 = pd.Series(sIC[:, 1] * A[0, 1], index=mixed_matrix.index)

        # add the global mean to ONE component
        c0 = c0 + mu1

        # write back
        sep1_n1.update(c0)
        if n_comp > 1:
            sep2_n1.update(c1)

        
        # baseline_seg = pd.Series(mu1, index=mixed_matrix.index)
        # base_n1.update(baseline_seg)


    elif k == 3:
        m = (cpu1.index >= pd.to_datetime(tL, unit='s')) & (cpu1.index < pd.to_datetime(tR, unit='s'))
        s = cpu1.loc[m]
        if s.empty:
            continue

        mixed_matrix = pd.DataFrame({
            'lag_0': s - mu1,
            'lag_1': s.shift(1) - mu1,
            'lag_2': s.shift(2) - mu1,
        }).dropna()

        if len(mixed_matrix) < 4+5:
            continue

        x = mixed_matrix.values
        rank = np.linalg.matrix_rank(x)
        n_comp = min(3, mixed_matrix.shape[1], rank)
        if n_comp < 1:
            continue

        ica = FastICA(n_components=n_comp, whiten='arbitrary-variance', random_state=42)
        sIC = ica.fit_transform(x)
        A = ica.mixing_

        c0 = pd.Series(sIC[:, 0] * A[0, 0], index=mixed_matrix.index); sep1_n1.update(c0)
        if n_comp > 1:
            c1 = pd.Series(sIC[:, 1] * A[0, 1], index=mixed_matrix.index); sep2_n1.update(c1)
        if n_comp > 2:
            c2 = pd.Series(sIC[:, 2] * A[0, 2], index=mixed_matrix.index); sep3_n1.update(c2)

        baseline_seg = pd.Series(mu1, index=mixed_matrix.index)  # global mean
        base_n1.update(baseline_seg)

# ============================
# Node 2
# ============================
for tL, tR, k in intervals_node2:
    if k == 2:
        m = (cpu2.index >= pd.to_datetime(tL, unit='s')) & (cpu2.index < pd.to_datetime(tR, unit='s'))
        s = cpu2.loc[m]
        if s.empty:
            continue

        mixed_matrix = pd.DataFrame({
            'lag_0': s - mu2,
            'lag_1': s.shift(1) - mu2,
        }).dropna()

        if len(mixed_matrix) < 3+5:
            continue

        x = mixed_matrix.values
        rank = np.linalg.matrix_rank(x)
        n_comp = min(2, mixed_matrix.shape[1], rank)
        if n_comp < 1:
            continue

        ica = FastICA(n_components=n_comp, whiten='arbitrary-variance', random_state=42)
        sIC = ica.fit_transform(x)
        A = ica.mixing_

        # ...
        c0 = pd.Series(sIC[:, 0] * A[0, 0], index=mixed_matrix.index)
        if n_comp > 1:
            c1 = pd.Series(sIC[:, 1] * A[0, 1], index=mixed_matrix.index)

        # >>> add the global mean to ONE component
        c0 = c0 + mu2

        # write back
        sep1_n2.update(c0)
        if n_comp > 1:
            sep2_n2.update(c1)

        # >>> do NOT update base_n2 here
        # baseline_seg = pd.Series(mu2, index=mixed_matrix.index)
        # base_n2.update(baseline_seg)


    elif k == 3:
        m = (cpu2.index >= pd.to_datetime(tL, unit='s')) & (cpu2.index < pd.to_datetime(tR, unit='s'))
        s = cpu2.loc[m]
        if s.empty:
            continue

        mixed_matrix = pd.DataFrame({
            'lag_0': s - mu2,
            'lag_1': s.shift(1) - mu2,
            'lag_2': s.shift(2) - mu2,
        }).dropna()

        if len(mixed_matrix) < 4+5:
            continue

        x = mixed_matrix.values
        rank = np.linalg.matrix_rank(x)
        n_comp = min(3, mixed_matrix.shape[1], rank)
        if n_comp < 1:
            continue

        ica = FastICA(n_components=n_comp, whiten='arbitrary-variance', random_state=42)
        sIC = ica.fit_transform(x)
        A = ica.mixing_

        c0 = pd.Series(sIC[:, 0] * A[0, 0], index=mixed_matrix.index); sep1_n2.update(c0)
        if n_comp > 1:
            c1 = pd.Series(sIC[:, 1] * A[0, 1], index=mixed_matrix.index); sep2_n2.update(c1)
        if n_comp > 2:
            c2 = pd.Series(sIC[:, 2] * A[0, 2], index=mixed_matrix.index); sep3_n2.update(c2)

        baseline_seg = pd.Series(mu2, index=mixed_matrix.index)  # global mean
        base_n2.update(baseline_seg)

# plots (unchanged)
sep1_n1_plot = sep1_n1.where(sep1_n1 != 0)
sep2_n1_plot = sep2_n1.where(sep2_n1 != 0)
sep3_n1_plot = sep3_n1.where(sep3_n1 != 0)

sep1_n2_plot = sep1_n2.where(sep1_n2 != 0)
sep2_n2_plot = sep2_n2.where(sep2_n2 != 0)
sep3_n2_plot = sep3_n2.where(sep3_n2 != 0)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
ax1, ax2 = axes

ax1.plot(cpu1.index, cpu1.values, label=f"{node1_name} total CPU", linewidth=2, color='black')
ax1.plot(sep1_n1_plot.index, sep1_n1_plot.values, label="Separated #1", alpha=0.9)
ax1.plot(sep2_n1_plot.index, sep2_n1_plot.values, label="Separated #2", alpha=0.9)
if not sep3_n1_plot.dropna().empty:
    ax1.plot(sep3_n1_plot.index, sep3_n1_plot.values, label="Separated #3", alpha=0.9)
ax1.set_title(f"Node: {node1_name}")
ax1.set_ylabel("CPU Utilization"); ax1.legend(loc="upper right"); ax1.grid(True, linestyle="--", alpha=0.3)

ax2.plot(cpu2.index, cpu2.values, label=f"{node2_name} total CPU", linewidth=2, color='black')
ax2.plot(sep1_n2_plot.index, sep1_n2_plot.values, label="Separated #1", alpha=0.9)
ax2.plot(sep2_n2_plot.index, sep2_n2_plot.values, label="Separated #2", alpha=0.9)
if not sep3_n2_plot.dropna().empty:
    ax2.plot(sep3_n2_plot.index, sep3_n2_plot.values, label="Separated #3", alpha=0.9)
ax2.set_title(f"Node: {node2_name}")
ax2.set_xlabel("Time (raw timestamps from data)")
ax2.set_ylabel("CPU Utilization"); ax2.legend(loc="upper right"); ax2.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig("zzzzzzzz11.png", dpi=300)
