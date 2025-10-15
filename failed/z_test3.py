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
for nd in node_list:
    if nd["node_name"] == node1_name:
        arr = nd["metrics"]["cpu_util"]  # list of [timestamp, util]
        t_n1 = np.array([float(x[0]) for x in arr])
        u_n1 = np.array([float(x[1]) for x in arr])
    elif nd["node_name"] == node2_name:
        arr = nd["metrics"]["cpu_util"]
        t_n2 = np.array([float(x[0]) for x in arr])
        u_n2 = np.array([float(x[1]) for x in arr])


# FASTICA on intervals with overlap 2 or 3
# y is array of time, y is array of utilization, time window, overlap count
# --- Drop-in: replace your run_interval_ica with this ---
# collectors
import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA

# ---- Build full time-indexed CPU series per node (change unit to 'ms' if needed)
cpu1 = pd.Series(u_n1, index=pd.to_datetime(t_n1, unit='s'), name='cpu')
cpu2 = pd.Series(u_n2, index=pd.to_datetime(t_n2, unit='s'), name='cpu')

# ---- Initialize separated outputs (zeros, same timeline)
sep1_n1 = pd.Series(0.0, index=cpu1.index)
sep2_n1 = pd.Series(0.0, index=cpu1.index)
sep3_n1 = pd.Series(0.0, index=cpu1.index)

sep1_n2 = pd.Series(0.0, index=cpu2.index)
sep2_n2 = pd.Series(0.0, index=cpu2.index)
sep3_n2 = pd.Series(0.0, index=cpu2.index)

# ============================
# Node 1: run per-interval ICA
# ============================
for tL, tR, k in intervals_node1:
    if k == 2:
        m = (cpu1.index >= pd.to_datetime(tL, unit='s')) & (cpu1.index < pd.to_datetime(tR, unit='s'))
        s = cpu1.loc[m]
        if s.empty: 
            continue

        mixed_matrix = pd.DataFrame({
            'lag_0': s,
            'lag_1': s.shift(1),
        }).dropna()

        if len(mixed_matrix) < 3:  # too short to be meaningful
            continue

                # --- clean and sanity-check the lag matrix before ICA ---
		# 1) make sure time is sorted & unique (safety)
        mixed_matrix = mixed_matrix.sort_index()
        mixed_matrix = mixed_matrix[~mixed_matrix.index.duplicated(keep='first')]

		# 2) drop any row with non-finite values (NaN/Inf)
        mixed_matrix = mixed_matrix.replace([np.inf, -np.inf], np.nan).dropna()
        if mixed_matrix.empty:
            continue

				# 3) drop near-constant columns (zero or tiny variance kills whitening/SVD)
        stds = mixed_matrix.std(ddof=0)
        keep_cols = stds[stds > 1e-8].index.tolist()
        mixed_matrix = mixed_matrix[keep_cols]
        if len(keep_cols) < k:      # not enough independent features for k components
            continue

				# 4) require enough samples for stability (tune the 10 as you like)
        if len(mixed_matrix) < max(10, 5 * k):
            continue

				# 5) de-mean and (optionally) standardize features to help numerics
        mixed_matrix = mixed_matrix - mixed_matrix.mean(axis=0)
        den = mixed_matrix.std(ddof=0).replace(0, 1.0)
        mixed_matrix = mixed_matrix / den

				# 6) final rank check; if rank < k, ICA will blow up
        if np.linalg.matrix_rank(mixed_matrix.values) < k:
            continue

        # --- standardize features and run ICA ---
        X = mixed_matrix.copy()
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0).replace(0, 1.0)
        X_std = (X - mu) / sigma

        ica = FastICA(n_components=2, whiten='arbitrary-variance', random_state=42)
        S = ica.fit_transform(X_std.values)
        A = ica.mixing_

        # --- reconstruct each component’s contribution to the original (lag_0) ---
        contrib = []
        for c in range(2):
            y_c = S[:, c] * A[0, c] * sigma['lag_0']
            contrib.append(pd.Series(y_c, index=X.index))

        baseline = pd.Series(mu['lag_0'], index=X.index)
        recon_lag0 = sum(contrib) + baseline  # optional: check vs s

# --- write back separated components ---
        sep1_n1.update(contrib[0])   # overwrites matching timestamps only
        sep2_n1.update(contrib[1])

    elif k == 3:
        m = (cpu1.index >= pd.to_datetime(tL, unit='s')) & (cpu1.index < pd.to_datetime(tR, unit='s'))
        s = cpu1.loc[m]
        if s.empty:
            continue

        mixed_matrix = pd.DataFrame({
            'lag_0': s,
            'lag_1': s.shift(1),
            'lag_2': s.shift(2),
        }).dropna()

        if len(mixed_matrix) < 4:
            continue

        # --- clean and sanity-check the lag matrix before ICA ---
				# 1) make sure time is sorted & unique (safety)
        mixed_matrix = mixed_matrix.sort_index()
        mixed_matrix = mixed_matrix[~mixed_matrix.index.duplicated(keep='first')]

				# 2) drop any row with non-finite values (NaN/Inf)
        mixed_matrix = mixed_matrix.replace([np.inf, -np.inf], np.nan).dropna()
        if mixed_matrix.empty:
            continue

				# 3) drop near-constant columns (zero or tiny variance kills whitening/SVD)
        stds = mixed_matrix.std(ddof=0)
        keep_cols = stds[stds > 1e-8].index.tolist()
        mixed_matrix = mixed_matrix[keep_cols]
        if len(keep_cols) < k:      # not enough independent features for k components
            continue

				# 4) require enough samples for stability (tune the 10 as you like)
        if len(mixed_matrix) < max(10, 5 * k):
            continue

				# 5) de-mean and (optionally) standardize features to help numerics
        mixed_matrix = mixed_matrix - mixed_matrix.mean(axis=0)
        den = mixed_matrix.std(ddof=0).replace(0, 1.0)
        mixed_matrix = mixed_matrix / den

				# 6) final rank check; if rank < k, ICA will blow up
        if np.linalg.matrix_rank(mixed_matrix.values) < k:
            continue

        X = mixed_matrix.copy()
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0).replace(0, 1.0)
        X_std = (X - mu) / sigma

        ica = FastICA(n_components=3, whiten='arbitrary-variance', random_state=42)
        S = ica.fit_transform(X_std.values)
        A = ica.mixing_

        contrib = []
        for c in range(3):
            y_c = S[:, c] * A[0, c] * sigma['lag_0']
            contrib.append(pd.Series(y_c, index=X.index))

        baseline = pd.Series(mu['lag_0'], index=X.index)
        recon_lag0 = sum(contrib) + baseline

        sep1_n1.update(contrib[0])
        sep2_n1.update(contrib[1])
        sep3_n1.update(contrib[2])

# ============================
# Node 2: run per-interval ICA
# ============================
for tL, tR, k in intervals_node2:
    if k == 2:
        m = (cpu2.index >= pd.to_datetime(tL, unit='s')) & (cpu2.index < pd.to_datetime(tR, unit='s'))
        s = cpu2.loc[m]
        if s.empty:
            continue

        mixed_matrix = pd.DataFrame({
            'lag_0': s,
            'lag_1': s.shift(1),
        }).dropna()

        if len(mixed_matrix) < 3:
            continue
            
        # --- clean and sanity-check the lag matrix before ICA ---
		# 1) make sure time is sorted & unique (safety)
        mixed_matrix = mixed_matrix.sort_index()
        mixed_matrix = mixed_matrix[~mixed_matrix.index.duplicated(keep='first')]

		# 2) drop any row with non-finite values (NaN/Inf)
        mixed_matrix = mixed_matrix.replace([np.inf, -np.inf], np.nan).dropna()
        if mixed_matrix.empty:
            continue

				# 3) drop near-constant columns (zero or tiny variance kills whitening/SVD)
        stds = mixed_matrix.std(ddof=0)
        keep_cols = stds[stds > 1e-8].index.tolist()
        mixed_matrix = mixed_matrix[keep_cols]
        if len(keep_cols) < k:      # not enough independent features for k components
            continue

				# 4) require enough samples for stability (tune the 10 as you like)
        if len(mixed_matrix) < max(10, 5 * k):
            continue

				# 5) de-mean and (optionally) standardize features to help numerics
        mixed_matrix = mixed_matrix - mixed_matrix.mean(axis=0)
        den = mixed_matrix.std(ddof=0).replace(0, 1.0)
        mixed_matrix = mixed_matrix / den

				# 6) final rank check; if rank < k, ICA will blow up
        if np.linalg.matrix_rank(mixed_matrix.values) < k:
            continue

        # --- standardize features and run ICA ---
        X = mixed_matrix.copy()
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0).replace(0, 1.0)
        X_std = (X - mu) / sigma

        ica = FastICA(n_components=2, whiten='arbitrary-variance', random_state=42)
        S = ica.fit_transform(X_std.values)
        A = ica.mixing_

        # --- reconstruct each component’s contribution to the original (lag_0) ---
        contrib = []
        for c in range(2):
            y_c = S[:, c] * A[0, c] * sigma['lag_0']
            contrib.append(pd.Series(y_c, index=X.index))

        baseline = pd.Series(mu['lag_0'], index=X.index)
        recon_lag0 = sum(contrib) + baseline  # optional: check vs s

# --- write back separated components ---
        sep1_n2.update(contrib[0])   # overwrites matching timestamps only
        sep2_n2.update(contrib[1])

    elif k == 3:
        m = (cpu2.index >= pd.to_datetime(tL, unit='s')) & (cpu2.index < pd.to_datetime(tR, unit='s'))
        s = cpu2.loc[m]
        if s.empty:
            continue

        mixed_matrix = pd.DataFrame({
            'lag_0': s,
            'lag_1': s.shift(1),
            'lag_2': s.shift(2),
        }).dropna()

        if len(mixed_matrix) < 4:
            continue
         
        # --- clean and sanity-check the lag matrix before ICA ---
				# 1) make sure time is sorted & unique (safety)
        mixed_matrix = mixed_matrix.sort_index()
        mixed_matrix = mixed_matrix[~mixed_matrix.index.duplicated(keep='first')]

				# 2) drop any row with non-finite values (NaN/Inf)
        mixed_matrix = mixed_matrix.replace([np.inf, -np.inf], np.nan).dropna()
        if mixed_matrix.empty:
            continue

				# 3) drop near-constant columns (zero or tiny variance kills whitening/SVD)
        stds = mixed_matrix.std(ddof=0)
        keep_cols = stds[stds > 1e-8].index.tolist()
        mixed_matrix = mixed_matrix[keep_cols]
        if len(keep_cols) < k:      # not enough independent features for k components
            continue

				# 4) require enough samples for stability (tune the 10 as you like)
        if len(mixed_matrix) < max(10, 5 * k):
            continue

				# 5) de-mean and (optionally) standardize features to help numerics
        mixed_matrix = mixed_matrix - mixed_matrix.mean(axis=0)
        den = mixed_matrix.std(ddof=0).replace(0, 1.0)
        mixed_matrix = mixed_matrix / den

				# 6) final rank check; if rank < k, ICA will blow up
        if np.linalg.matrix_rank(mixed_matrix.values) < k:
            continue

        X = mixed_matrix.copy()
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0).replace(0, 1.0)
        X_std = (X - mu) / sigma

        ica = FastICA(n_components=3, whiten='arbitrary-variance', random_state=42)
        S = ica.fit_transform(X_std.values)
        A = ica.mixing_

        contrib = []
        for c in range(3):
            y_c = S[:, c] * A[0, c] * sigma['lag_0']
            contrib.append(pd.Series(y_c, index=X.index))

        baseline = pd.Series(mu['lag_0'], index=X.index)
        recon_lag0 = sum(contrib) + baseline

        sep1_n2.update(contrib[0])
        sep2_n2.update(contrib[1])
        sep3_n2.update(contrib[2])

# After this:
# - intervals_node1 / intervals_node2 hold (t_left, t_right, overlap_count)
# - results_node1 / results_node2 are lists of dicts with:
#   interval, k, cpu_segment_time, cpu_segment, components (k arrays), components_sum, reconstruction, mse_vs_cpu, mae_vs_cpuu
# mask zeros so the separated parts appear only where active
sep1_n1_plot = sep1_n1.where(sep1_n1 != 0)
sep2_n1_plot = sep2_n1.where(sep2_n1 != 0)
sep3_n1_plot = sep3_n1.where(sep3_n1 != 0)

sep1_n2_plot = sep1_n2.where(sep1_n2 != 0)
sep2_n2_plot = sep2_n2.where(sep2_n2 != 0)
sep3_n2_plot = sep3_n2.where(sep3_n2 != 0)

# --- Plot setup ---
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
ax1, ax2 = axes

# ==================== NODE 1 ====================
ax1.plot(cpu1.index, cpu1.values, label=f"{node1_name} total CPU", linewidth=2, color='black')
ax1.plot(sep1_n1_plot.index, sep1_n1_plot.values, label="Separated #1", alpha=0.9)
ax1.plot(sep2_n1_plot.index, sep2_n1_plot.values, label="Separated #2", alpha=0.9)
if not sep3_n1_plot.dropna().empty:
    ax1.plot(sep3_n1_plot.index, sep3_n1_plot.values, label="Separated #3", alpha=0.9)

ax1.set_title(f"Node: {node1_name}")
ax1.set_ylabel("CPU Utilization")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.3)

# ==================== NODE 2 ====================
ax2.plot(cpu2.index, cpu2.values, label=f"{node2_name} total CPU", linewidth=2, color='black')
ax2.plot(sep1_n2_plot.index, sep1_n2_plot.values, label="Separated #1", alpha=0.9)
ax2.plot(sep2_n2_plot.index, sep2_n2_plot.values, label="Separated #2", alpha=0.9)
if not sep3_n2_plot.dropna().empty:
    ax2.plot(sep3_n2_plot.index, sep3_n2_plot.values, label="Separated #3", alpha=0.9)

ax2.set_title(f"Node: {node2_name}")
ax2.set_xlabel("Time (raw timestamps from data)")
ax2.set_ylabel("CPU Utilization")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig("zzz11.png", dpi=300)