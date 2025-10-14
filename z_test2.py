# FASTICA algorithm.py  (procedural, no extra libraries, no functions)
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA

# ------------------------- CONFIG (edit as needed) -------------------------
workloads_json = "all_workloads_ic2.json"
sysloads_json  = "all_system_loads_ic2.json"
workload_idx   = 16              # choose which workload to analyze
node_choice    = None            # set a node name to force (e.g., "amphost001"); None = auto-pick first with data
max_jobs_shown = 3               # we mirror your original s1/s2/s3 idea

# ------------------------- LOAD DATA -------------------------
with open(workloads_json, "r") as f:
    workloads = json.load(f)
with open(sysloads_json, "r") as f:
    sysloads = json.load(f)

workload = workloads[workload_idx]
tasks = workload["tasklist"]

sys_entry = sysloads[workload_idx]
node_list = sys_entry["node_list"]

# Build per-node time series (cpu util vs time). Metrics may be [[ts,val], ...] OR list of dicts.
node_to_series = {}
for node in node_list:
    name = node.get("node_name") or node.get("name")
    raw = node.get("metrics") or node.get("cpu_util") or node.get("data") or []
    rows = []
    if isinstance(raw, list):
        for m in raw:
            if isinstance(m, (list, tuple)) and len(m) >= 2:
                try:
                    ts = float(m[0]); val = float(m[1]); rows.append([ts, val])
                except Exception:
                    pass
            elif isinstance(m, dict):
                ts = m.get("timestamp", m.get("time", m.get("ts", m.get("t", None))))
                val = m.get("cpu_util", m.get("value", m.get("usage", m.get("cpu", m.get("y", None)))))
                try:
                    if ts is not None and val is not None:
                        tsf = float(ts); vf = float(val); rows.append([tsf, vf])
                except Exception:
                    pass
    arr = np.asarray(rows, dtype=float) if rows else np.empty((0,2), dtype=float)
    if arr.size == 0:
        node_to_series[name] = pd.Series(dtype=float)
    else:
        ts = pd.to_datetime(arr[:,0], unit="s", utc=True, errors="coerce")
        util = arr[:,1].astype(float)
        mask = ~pd.isna(ts)
        node_to_series[name] = pd.Series(util[mask], index=ts[mask]).sort_index()

# Build per-node job intervals: node_name -> list of (task_id, start_ts, end_ts)
node_to_jobs = {}
for node in node_to_series.keys():
    node_to_jobs[node] = []
for task in tasks:
    try:
        t_id = str(task["task_id"])
        start = pd.to_datetime(float(task["start_time"]), unit="s", utc=True)
        end   = pd.to_datetime(float(task["finish_time"]), unit="s", utc=True)
        for n in task.get("nodes", []):
            nn = n["node_name"]
            if nn in node_to_jobs:
                node_to_jobs[nn].append((t_id, start, end))
    except Exception:
        pass

# ------------------------- PICK NODE -------------------------
chosen_node = None
for name, s in node_to_series.items():
    if node_choice and name == node_choice and not s.empty and len(node_to_jobs.get(name, [])) > 0:
        chosen_node = name
        break
for name, s in node_to_series.items():
    if chosen_node is None and not s.empty and len(node_to_jobs.get(name, [])) > 0:
        chosen_node = name
        break

if chosen_node is None:
    raise RuntimeError("No node with both metrics and jobs found in this workload index.")

total_signal = node_to_series[chosen_node]
jobs_on_node = node_to_jobs[chosen_node]

# ------------------------- BUILD ELEMENTARY INTERVALS -------------------------
boundaries = set()
for jid, s, e in jobs_on_node:
    boundaries.add(s); boundaries.add(e)
boundaries = sorted(boundaries)
intervals = []
for i in range(len(boundaries)-1):
    t0 = boundaries[i]; t1 = boundaries[i+1]
    if (total_signal.index >= t0).any() and (total_signal.index < t1).any():
        intervals.append((t0, t1))

# ------------------------- CONSTRUCT "ACTIVE" MASKS PER JOB -------------------------
# Build a unified time index covering the node's available samples
dates = total_signal.index
n_samples = len(dates)
job_ids_sorted = sorted(list({jid for jid,_,_ in jobs_on_node}))
job_to_mask = {}
for jid in job_ids_sorted:
    mask = pd.Series(False, index=dates)
    for j2, s, e in jobs_on_node:
        if j2 == jid:
            mask |= ((dates >= s) & (dates < e))
    job_to_mask[jid] = mask

# For plotting like your original (s1/s2/s3)
s1_separated = pd.Series(np.zeros(n_samples), index=dates)
s2_separated = pd.Series(np.zeros(n_samples), index=dates)
s3_separated = pd.Series(np.zeros(n_samples), index=dates)

# ------------------------- MAIN LOOP: HANDLE EACH INTERVAL -------------------------
for (t0, t1) in intervals:
    # samples in this elementary interval
    mask = (dates >= t0) & (dates < t1)
    if not mask.any():
        continue
    y = total_signal.loc[mask]

    # which jobs are active here?
    active_jobs = []
    for jid in job_ids_sorted:
        if job_to_mask[jid].loc[mask].any():
            # overlap if job spans this interval (strictly start < t1 and end > t0)
            # We already built masks; just check any True in this window.
            active_jobs.append(jid)

    k = len(active_jobs)

    # ---- 1-overlap: assign directly
    if k == 1:
        jid = active_jobs[0]
        if max_jobs_shown >= 1:
            if jid == job_ids_sorted[0]:
                s1_separated.loc[mask] += y.values
            elif len(job_ids_sorted) > 1 and jid == job_ids_sorted[1] and max_jobs_shown >= 2:
                s2_separated.loc[mask] += y.values
            elif len(job_ids_sorted) > 2 and jid == job_ids_sorted[2] and max_jobs_shown >= 3:
                s3_separated.loc[mask] += y.values

    # ---- 2- or 3-overlap: FastICA with lag embedding; ensure parts sum to y
    elif k in (2, 3):
        n_sources = k

        # time-delay embedding with exactly n_sources lags (lag_0..lag_{K-1})
        X_df = pd.DataFrame({f"lag_{i}": y.shift(i) for i in range(n_sources)}).dropna()
        if X_df.empty:
            # not enough points for embedding; equal split fallback to preserve add-back
            parts_mat = np.tile((y.values / n_sources)[:len(y)], (n_sources,1))
            parts_idx = y.index
        else:
            idx_local = X_df.index
            X = X_df.values.astype(np.float64)

            # FastICA
            ica = FastICA(n_components=n_sources, whiten="arbitrary-variance", random_state=0)
            S = ica.fit_transform(X)    # (T, K)
            A = ica.mixing_             # (K, K)
            mean_vec = ica.mean_        # (K,)

            # each component's contribution on lag_0 channel
            contribs = []
            for comp in range(n_sources):
                contribs.append(S[:, comp] * A[0, comp])
            contribs = np.column_stack(contribs)   # (T, K)

            # reconstructed lag_0
            recon = np.sum(contribs, axis=1) + float(mean_vec[0])   # (T,)

            # distribute tiny residual drift so parts sum to the observed exactly
            y_aligned = y.loc[idx_local].values
            drift = y_aligned - recon  # (T,)
            # energy weights per component
            energy = np.maximum(1e-12, np.mean(contribs**2, axis=0))  # (K,)
            weights = energy / np.sum(energy)
            for comp in range(n_sources):
                contribs[:, comp] = contribs[:, comp] + weights[comp] * drift

            # final parts (point-wise sum == y on idx_local)
            parts_mat = contribs.T                       # (K, T)
            parts_idx = idx_local

        # write parts back to s1/s2/s3 according to active job order (sorted by task_id)
        # map first active job -> s1, second -> s2, third -> s3 (only if exists to show)
        for jpos in range(n_sources):
            if jpos == 0 and max_jobs_shown >= 1:
                s1_separated.loc[parts_idx] += parts_mat[0]
            elif jpos == 1 and max_jobs_shown >= 2:
                s2_separated.loc[parts_idx] += parts_mat[1]
            elif jpos == 2 and max_jobs_shown >= 3:
                s3_separated.loc[parts_idx] += parts_mat[2]

    else:
        # >3 overlaps: equal split to preserve add-back without introducing extra libs
        if k > 0:
            share = y.values / float(k)
            for jpos in range(min(k, max_jobs_shown)):
                if jpos == 0:
                    s1_separated.loc[mask] += share
                elif jpos == 1:
                    s2_separated.loc[mask] += share
                elif jpos == 2:
                    s3_separated.loc[mask] += share

# ------------------------- PLOT (mirror your original style) -------------------------
plt.figure(figsize=(14, 8))

plt.subplot(4,1,1)
plt.title(f"Mixed Signal — Node {chosen_node} (Workload {workload_idx})")
plt.plot(total_signal.index, total_signal.values, label="Mixed Signal", color='black', linewidth=1.2)
plt.legend(); plt.grid(True)

plt.subplot(4,1,2)
plt.title("Separated Source 1")
plt.plot(s1_separated.index, s1_separated.values, label="Separated Source 1", linestyle='--')
plt.legend(); plt.grid(True)

plt.subplot(4,1,3)
plt.title("Separated Source 2")
plt.plot(s2_separated.index, s2_separated.values, label="Separated Source 2", linestyle='--')
plt.legend(); plt.grid(True)

plt.subplot(4,1,4)
plt.title("Separated Source 3")
plt.plot(s3_separated.index, s3_separated.values, label="Separated Source 3", linestyle='--')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()