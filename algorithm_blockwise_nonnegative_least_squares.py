"""
Hybrid mean + blockwise NNLS workload decomposition
---------------------------------------------------
For each workload and node:
  1. Mean-only greedy initialization
  2. Blockwise regularized NNLS refinement
Outputs:
  /mnt/data/mean_contribs/
    ├── per_node_task_contribs_<workload>.npy
    └── plots/<workload>/<node>.png
"""

import json, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

# ---------- Configuration ----------
DATA_DIR = Path("./")
OUT_DIR = DATA_DIR / "results_blockwise_NNLS"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

BLOCK_SIZE = 30.0   # seconds per block
L2 = 1e-2           # regularization weight

# ---------- Helpers ----------
def ts_float(x): return float(x) if not isinstance(x, (list, tuple)) else float(x[0])

def plot_node(ts, obs, recon, path):
    plt.figure(figsize=(8,3))
    plt.plot(ts, obs, label="Observed", lw=1.2)
    plt.plot(ts, recon, label="Reconstructed", lw=1)
    plt.legend(); plt.xlabel("Time"); plt.ylabel("Util")
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120); plt.close()

def plot_tasks(ts, obs, contribs, path):
    plt.figure(figsize=(8,3))
    #plt.plot(ts, obs, label="Observed", lw=1.2)
    for tid, arr in contribs.items():
        plt.plot(ts, arr, label = f"Task {tid}", alpha = 0.8)
    #plt.plot(ts, recon, label="Reconstructed", lw=1)
    plt.legend(); plt.xlabel("Time"); plt.ylabel("Util")
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120); plt.close()
# ---------- Core hybrid algorithm ----------
def mean_then_blockwise_nnls(ts, y, tasks, block_size=30, l2=1e-2):
    T, N = len(ts), len(tasks)
    contribs = {t["task_id"]: np.zeros(T) for t in tasks}
    '''
    # --- Mean-only greedy init ---
    active = np.zeros((N, T), bool)
    for j,t in enumerate(tasks):
        active[j] = (ts >= t["start"]) & (ts < t["finish"])
    residual, known = y.copy(), np.zeros(N,bool)
    for _ in range(20):
        progress=False
        s = active.sum(0)
        for j in range(N):
            if known[j]: continue
            mask_only = active[j] & (s==1)
            if not mask_only.any():
                mask_ok = active[j].copy()
                for k in range(N):
                    if k==j or known[k]: continue
                    mask_ok &= ~active[k]
                mask=mask_ok
            else: mask=mask_only
            if mask.any():
                est=max(0,np.nanmean(residual[mask]))
                contribs[tasks[j]["task_id"]] = active[j].astype(float)*est
                residual -= contribs[tasks[j]["task_id"]]
                known[j]=True; progress=True
        if not progress: break
    '''
    # --- Blockwise NNLS refinement ---
    A_cols, ids = [], []
    for t in tasks:
        start,fin=t["start"],t["finish"]
        nb=max(1,int(np.ceil((fin-start)/block_size)))
        edges=np.linspace(start,fin,nb+1)
        for b in range(nb):
            m=(ts>=edges[b])&(ts<edges[b+1])
            A_cols.append(m.astype(float)); ids.append((t["task_id"],b))
    if not A_cols: return contribs
    A=np.vstack(A_cols).T; A_aug=np.vstack([A,np.sqrt(l2)*np.eye(A.shape[1])])
    y_aug=np.concatenate([y,np.zeros(A.shape[1])])
    res=lsq_linear(A_aug,y_aug,bounds=(0,np.inf)); x=res.x
    for k in contribs: contribs[k][:]=0
    for c,(tid,_) in enumerate(ids): contribs[tid]+=A[:,c]*x[c]
    return contribs

# ---------- Load data ----------
system_all = json.load(open(DATA_DIR/"all_system_loads_ic2.json"))
workloads_all = json.load(open(DATA_DIR/"all_workloads_ic2.json"))
assert len(system_all) == len(workloads_all), "System/workload list lengths differ!"

print(f"Loaded {len(system_all)} parallel workload+system pairs.")

# ---------- Loop over pairs ----------
for wi, (system_entry, workload_entry) in enumerate(zip(system_all, workloads_all)):
    wname = workload_entry.get("workload_name") or workload_entry.get("name") or f"w{wi}"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in wname)
    plotdir = PLOTS_DIR / safe
    plotdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Processing {wname} (pair {wi}) ===")

    # Build node utilization for this workload
    node_series = {}
    for node in system_entry.get("node_list", []):
        name = node.get("node_name")
        pairs = node.get("metrics", {}).get("cpu_util", [])
        if not pairs: continue
        t, v = zip(*pairs)
        ts = np.array(list(map(float, t)))
        vals = np.array(list(map(float, v)))
        order = np.argsort(ts)
        node_series[name] = {
            "timestamps": ts[order],
            "util": vals[order]
        }

    # Build task list for this workload
    tasks = [{
        "task_id": int(t["task_id"]),
        "start": float(t.get("start_time", t.get("submit_time", 0))),
        "finish": float(t.get("finish_time", 0)),
        "nodes": [n["node_name"] for n in t.get("nodes", []) if n.get("node_name")]
    } for t in workload_entry.get("tasklist", [])]

    # Run hybrid algorithm for each node
    contribs_all = {}
    for node, data in node_series.items():
        node_tasks = [t for t in tasks if node in t["nodes"]]
        if not node_tasks: continue
        contribs = mean_then_blockwise_nnls(
            data["timestamps"], data["util"], node_tasks, BLOCK_SIZE, L2
        )
        recon = sum(contribs.values())
        

        plot_tasks(data["timestamps"],data["util"], contribs,plotdir / f"task_{node}.png")
        plot_node(data["timestamps"], data["util"], recon, plotdir / f"{node}.png")
        contribs_all[node] = contribs

    np.save(OUT_DIR / f"per_node_task_contribs_{safe}.npy",
            {"workload": wname, "contribs": contribs_all}, allow_pickle=True)
    print(f"Saved per_node_task_contribs_{safe}.npy")
