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
    #plt.savefig(path, dpi=120); plt.close()

def plot_tasks(ts, obs, contribs, path):
    plt.figure(figsize=(8,3))
    #plt.plot(ts, obs, label="Observed", lw=1.2)
    for tid, arr in contribs.items():
        plt.plot(ts, arr, label = f"Task {tid}", alpha = 0.8)
    #plt.plot(ts, recon, label="Reconstructed", lw=1)
    plt.legend(); plt.xlabel("Time"); plt.ylabel("Util")
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    #plt.savefig(path, dpi=120); plt.close()


# ---------- Core hybrid algorithm ----------
def mean_then_blockwise_nnls(ts, y, tasks, block_size=30, l2=1e-2): #time, util, task
    T, N = len(ts), len(tasks)
    # T: no_of_timestamps, N: no_of_tasks
    contribs = {t["task_id"]: np.zeros(T) for t in tasks}
    # initialize to all zeros for every task

    # --- Blockwise NNLS refinement ---
    A_cols, ids = [], [] #A_cols: list of columns of A; ids: list of (task_id, block_id) pairs

    for t in tasks: #loop through each task dictionary
        start,fin=t["start"],t["finish"] #get start and finish time of the task
        nb=max(1,int(np.ceil((fin-start)/block_size))) #number of blocks for this task
        edges=np.linspace(start,fin,nb+1) #block boundaries
        for b in range(nb): #loop through each block
            m=(ts>=edges[b])&(ts<edges[b+1]) #mask for timestamps in this block
            A_cols.append(m.astype(float)); ids.append((t["task_id"],b))
            #convert mask to 0/1, append the column to A_cols
            #record which task and which block this column corresponds to
    if not A_cols: return contribs #no tasks case return zero (edge case)
    A=np.vstack(A_cols).T; A_aug=np.vstack([A,np.sqrt(l2)*np.eye(A.shape[1])])
    #A: design matrix with shape (T, total_blocks) T is at line 52
    #each column is one task block activity window
    # A_aug is A with L2 regularization rows added
    y_aug=np.concatenate([y,np.zeros(A.shape[1])])
    #extend target vector to match augmented system
    res=lsq_linear(A_aug,y_aug,bounds=(0,np.inf)); x=res.x
    #solve NNLS problem to get block contributions
    for k in contribs: contribs[k][:]=0
    #reset all contributions to zero
    for c,(tid,_) in enumerate(ids): #loop through each column/block
        contribs[tid]+=A[:,c]*x[c] #add block contribution to corresponding task
        """
        per task utilization
        """
    return contribs
    """
    contribs = {task_id: utilization_time_series}
    """


def pertask_utilization_NNLS():

    system_all = json.load(open(DATA_DIR/"all_system_loads_ic2.json"))
    workloads_all = json.load(open(DATA_DIR/"all_workloads_ic2.json"))
    assert len(system_all) == len(workloads_all)

    contribs_all = {} #initialize empty dictionary to hold all contributions

    for wi, (system_entry, workload_entry) in enumerate(zip(system_all, workloads_all)):
        wname = workload_entry.get("workload_name") or workload_entry.get("name") or f"w{wi}"
        #get the workload name
        #print(wname)

        node_series = {} #initialize empty dictionary to hold node time series
        for node in system_entry.get("node_list", []):
            name = node.get("node_name") #get node name
            #print(name)
            pairs = node.get("metrics", {}).get("cpu_util", []) #get list of (timestamp, util) tuples
            if not pairs: continue #skip if no data
            t, v = zip(*pairs) #unpack timestamps and utilizations
            ts = np.array(list(map(float, t))) #convert timestamps to float numpy array
            vals = np.array(list(map(float, v))) #same to utilizations
            order = np.argsort(ts) #sort timestamps

            node_series[name] = {
                "timestamps": ts[order],
                "util": vals[order]
            }

        #get task data 
        tasks = [{
            "task_id": int(t["task_id"]),
            "start": float(t.get("start_time", t.get("submit_time", 0))),
            "finish": float(t.get("finish_time", 0)),
            "nodes": [n["node_name"] for n in t.get("nodes", []) if n.get("node_name")]
        } for t in workload_entry.get("tasklist", [])]

        
        for node, data in node_series.items(): #loop through each node's time series data
            node_tasks = [t for t in tasks if node in t["nodes"]] #get tasks that ran on this node
            if not node_tasks: continue #skip if no tasks on this node

            contribs = mean_then_blockwise_nnls(
                data["timestamps"], data["util"], node_tasks, BLOCK_SIZE, L2
            ) #run hybrid algorithm to get per-task contributions
            
            ts = data["timestamps"]
            active_contribs = []

            for tid, util in contribs.items(): #contribs is task_id -> utilization time series
                mask = util > 0 #find timestamps where task had non-zero contribution
                if not np.any(mask): #skip if task had no contribution
                    continue

                active_contribs.append({
                    "task_id": tid,
                    "Node_name": node,
                    "Util": list(zip(ts[mask].tolist(), util[mask].tolist()))
                }) #store task_id, node name, and non-zero utilization time series
                
                """
                Util = [(timestamp, utilization), ...]
                """

                #print(active_contribs)
                #break

            if active_contribs:
                contribs_all[(wname, node)] = active_contribs
                
            #print(wi)

            #print(contribs_all)
    return contribs_all


    





def nnls_main(workloads_all, system_all, utilization):
    # ---------- Load data ----------
    #system_all = json.load(open(DATA_DIR/system_file))
    #workloads_all = json.load(open(DATA_DIR/workload_file))
    assert len(system_all) == len(workloads_all), "System/workload list lengths differ!"

    print(f"Loaded {len(system_all)} parallel workload+system pairs.")
    all_MSE = 0
    all_MAPE = 0
    all_MAE = 0
    MAE_list = []
    # ---------- Loop over pairs ----------
    for wi, (system_entry, workload_entry) in enumerate(zip(system_all, workloads_all)):
        wname = workload_entry.get("workload_name") or workload_entry.get("name") or f"w{wi}"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in wname)
        plotdir = PLOTS_DIR / safe
        plotdir.mkdir(parents=True, exist_ok=True)
        #print(f"\n=== Processing {wname} (pair {wi}) ===")

        # Build node utilization for this workload
        node_series = {}
        for node in system_entry.get("node_list", []):
            name = node.get("node_name")
            pairs = node.get("metrics", {}).get(f"{utilization}_util", [])
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
        c = 0.0
        single_MSE = 0.0
        single_MAPE = 0.0
        single_MAE = 0.0
        contribs_all = {}
        for node, data in node_series.items():
            node_abs_sum = 0.0
            t_count = 0
            node_tasks = [t for t in tasks if node in t["nodes"]]
            if not node_tasks: continue
            contribs = mean_then_blockwise_nnls(
                data["timestamps"], data["util"], node_tasks, BLOCK_SIZE, L2
            )

            recon = sum(contribs.values())
            #sum all task contributions to get reconstructed node utilization

            utilization_data = data["util"]
            timestamp = data["timestamps"]

            a = 0
            b = 0
            for reconstructed, original, t in zip(recon, utilization_data, timestamp):
                
                #print(reconstructed, original)
                if (reconstructed - original) < 0 or original == 0:
                    continue
                    
                #print("C", c)  
                skip = False
                #print(abs(reconstructed - original)/original*100)  
                for task in tasks:
                    if int(t) in [int(task["start"]), int(task["start"])-1,int(task["start"])+1] or int(t) in [int(task["finish"]),int(task["finish"])-1,int(task["finish"])+1]:
                       #print("found")
                       skip = True
                       continue
                if skip:
                    continue
                '''
                if (reconstructed - original)**2 > 25:
                    for task in tasks:
                        if int(t) in [int(task["start"]), int(task["start"])-1,int(task["start"])+1] or int(t) in [int(task["finish"]),int(task["finish"])-1,int(task["finish"])+1]:
                            #print("found")
                            skip = True
                            continue
                    if skip:
                        continue

                        
                        #else:
                            #print(f"time: {t}, task id:{task['task_id']}, start:{task['start']}, finish:{task['finish']}")

                if abs(reconstructed - original)/original*100 > 25:
                    for task in tasks:
                        if int(t) in [int(task["start"]), int(task["start"])-1,int(task["start"])+1] or int(t) in [int(task["finish"]),int(task["finish"])-1,int(task["finish"])+1]:
                            #print("found")
                            skip = True
                            #b += 1
                            continue
                    if skip:
                        continue
                #print(abs(reconstructed - original)/original*100)
                #a += 1


                        #else:
                            #print(f"time: {t}, task id:{task['task_id']}, start:{task['start']}, finish:{task['finish']}")
                '''
                single_MSE += (reconstructed - original) ** 2
                single_MAPE += abs(reconstructed - original)/original*100
                single_MAE += abs(reconstructed - original)
                node_abs_sum += abs(reconstructed - original)
                t_count += 1
                c += 1.0
                
        
        

            #plot_tasks(data["timestamps"],data["util"], contribs,plotdir / f"task_{node}.png")
            #plot_node(data["timestamps"], data["util"], recon, plotdir / f"{node}.png")
            contribs_all[node] = contribs
            if t_count > 0:
                #print(t_count)
                MAE_list.append(float(node_abs_sum/t_count))
            #else:
                #print(wi)

        np.save(OUT_DIR / f"per_node_task_contribs_{safe}.npy",
                {"workload": wname, "contribs": contribs_all}, allow_pickle=True)
        #print(f"Saved per_node_task_contribs_{safe}.npy")

        MSE = single_MSE/c
        MAPE = single_MAPE/c
        MAE = single_MAE/c
        #print(f"MSE: {MSE}, MAPE: {MAPE}, MAE: {MAE}")
        #print(f"MAE: {MAE}")
        all_MSE += MSE
        all_MAPE += MAPE
        all_MAE += MAE
        #MAE_list.append(float(MAE))
        

    all_MSE /= len(system_all)
    all_MAPE /= len(system_all)
    all_MAE /= len(system_all)
    #print(f"\n=== Overall Results ===\nMSE: {all_MSE}, MAPE: {all_MAPE}, MAE: {all_MAE}")
    #print(f"\n=== Overall Results ===\nMAE: {all_MAE}")
    #print(a, b)
    #print("MAE list:", MAE_list)
    
    #print(f"NNLS_{utilization}: {len(MAE_list)}")
    return MAE_list

#pertask_utilization_NNLS()
#main()