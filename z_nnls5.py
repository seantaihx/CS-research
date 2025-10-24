# -----------------------------------------imports -----------------------------------
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# --------------------------------------- load data --------------------------------
with open("all_workloads_ic2.json", "r") as f:
    workloads = json.load(f)
with open("all_system_loads_ic2.json", "r") as f:
    sysloads = json.load(f)

# ----------------------------------- run all workloads ----------------------------------
for i in range(25):
    workload = workloads[i]
    tasklist = workload["tasklist"]
    sys_entry = sysloads[i]
    node_list = sys_entry["node_list"]

    # ---------------------------- get two node names ---------------------------------
    node_names = []
    seen_nodes = set()
    for t in tasklist:
        for nd in t["nodes"]:
            n = nd["node_name"]
            if n not in seen_nodes:
                seen_nodes.add(n)
                node_names.append(n)
            if len(node_names) == 2:
                break
        if len(node_names) == 2:
            break
    if len(node_names) < 2:
        print(f"[Workload {i}] <2 nodes found. Skipping.")
        continue
    node1_name, node2_name = node_names[0], node_names[1]

    # --------------------------------- pull CPU time series ----------------------
    for nd in node_list:
        if nd["node_name"] == node1_name:
            arr = nd["metrics"]["cpu_util"]
            t_n1 = np.array([float(x[0]) for x in arr], dtype=float)
            u_n1 = np.array([float(x[1]) for x in arr], dtype=float)
        elif nd["node_name"] == node2_name:
            arr = nd["metrics"]["cpu_util"]
            t_n2 = np.array([float(x[0]) for x in arr], dtype=float)
            u_n2 = np.array([float(x[1]) for x in arr], dtype=float)

    idx1 = pd.to_datetime(t_n1, unit='s')
    idx2 = pd.to_datetime(t_n2, unit='s')
    cpu1 = pd.Series(u_n1, index=idx1)
    cpu2 = pd.Series(u_n2, index=idx2)

    # ---------------------------- collect tasks per node (start, end) ---------------------
    tasks_node1 = []   # (task_idx, start, end)
    tasks_node2 = []

    for t_idx, t in enumerate(tasklist):
        s = float(t["start_time"])
        e = float(t["finish_time"])
        for nd in t["nodes"]:
            n = nd["node_name"]
            if n == node1_name:
                tasks_node1.append((len(tasks_node1), s, e))
            elif n == node2_name:
                tasks_node2.append((len(tasks_node2), s, e))


    cols1 = [f"Task{str(k+1).zfill(2)}" for k in range(50)] #for plotting
    cols2 = [f"Task{str(k+1).zfill(2)}" for k in range(50)]

    # build A and solve NNLS per node with blockwise BCD 
    # Ax = b
    # Node 1
    T1 = len(cpu1.index); n1 = len(tasks_node1)
    A1 = np.zeros((T1, n1), dtype=float)
    for (j, s, e) in tasks_node1:
        m = (cpu1.index.view('i8')/1e9 >= s) & (cpu1.index.view('i8')/1e9 < e)
        A1[np.nonzero(m)[0], j] = 1.0
    b1 = cpu1.values.astype(float)

'''
Timestamps → rows
Tasks → columns

A1 =
[[1, 0, 0, 1],   # time 0s
 [1, 1, 0, 0],   # time 10s
 [0, 1, 1, 0],   # time 20s
 ... ]
'''

    # Node 2
    T2 = len(cpu2.index); n2 = len(tasks_node2)
    A2 = np.zeros((T2, n2), dtype=float)
    for (j, s, e) in tasks_node2:
        m = (cpu2.index.view('i8')/1e9 >= s) & (cpu2.index.view('i8')/1e9 < e)
        A2[np.nonzero(m)[0], j] = 1.0
    b2 = cpu2.values.astype(float)

    # ---------- blockwise NNLS (projected gradient on each block) ----------
    # settings
    max_outer = 40       # outer passes over all blocks
    inner_iters = 60     # PGD steps per block update
    block_size = 10      # number of task columns per block
    eps = 1e-8

    # Node 1 solve
    if n1 > 0:
        x1 = np.zeros(n1, dtype=float)   # task intensities
        y1 = A1.dot(x1)                  # current reconstruction
        blocks1 = [np.arange(s, min(s+block_size, n1)) for s in range(0, n1, block_size)]
        for _ in range(max_outer):
            for blk in blocks1:
                Ablk = A1[:, blk]               # (T1 x nb)
                xblk = x1[blk].copy()           # (nb,)
                # remove current block contr from y1
                y1 -= Ablk.dot(xblk)
                r = b1 - y1                      # residual for this block
                # PGD on min ||Ablk xblk - r||^2, xblk>=0
                step = 1.0 / (np.linalg.norm(Ablk, 'fro')**2 + eps)
                for __ in range(inner_iters):
                    g = Ablk.T.dot(Ablk.dot(xblk) - r)
                    xblk -= step * g
                    xblk = np.maximum(0.0, xblk)
                # write back
                y1 += Ablk.dot(xblk)
                x1[blk] = xblk
        recon1 = y1
    else:
        x1 = np.zeros(0, dtype=float)
        recon1 = np.zeros(T1, dtype=float)

    # Node 2 solve
    if n2 > 0:
        x2 = np.zeros(n2, dtype=float)
        y2 = A2.dot(x2)
        blocks2 = [np.arange(s, min(s+block_size, n2)) for s in range(0, n2, block_size)]
        for _ in range(max_outer):
            for blk in blocks2:
                Ablk = A2[:, blk]
                xblk = x2[blk].copy()
                y2 -= Ablk.dot(xblk)
                r = b2 - y2
                step = 1.0 / (np.linalg.norm(Ablk, 'fro')**2 + eps)
                for __ in range(inner_iters):
                    g = Ablk.T.dot(Ablk.dot(xblk) - r)
                    xblk -= step * g
                    xblk = np.maximum(0.0, xblk)
                y2 += Ablk.dot(xblk)
                x2[blk] = xblk
        recon2 = y2
    else:
        x2 = np.zeros(0, dtype=float)
        recon2 = np.zeros(T2, dtype=float)

    # ---------- per-task time series (A * diag(x)) ----------
    df1 = pd.DataFrame(0.0, index=cpu1.index, columns=cols1)
    for idx in range(n1):
        df1.iloc[:, idx] = A1[:, idx] * x1[idx]
    df2 = pd.DataFrame(0.0, index=cpu2.index, columns=cols2)
    for idx in range(n2):
        df2.iloc[:, idx] = A2[:, idx] * x2[idx]

    # ---------- metrics ----------
    mse1 = float(np.mean((recon1 - b1)**2)) if T1 > 0 else float('nan')
    mse2 = float(np.mean((recon2 - b2)**2)) if T2 > 0 else float('nan')

    m1 = b1 != 0
    m2 = b2 != 0
    mape1 = float(np.mean(np.abs((recon1[m1] - b1[m1]) / b1[m1])) * 100) if m1.any() else float('nan')
    mape2 = float(np.mean(np.abs((recon2[m2] - b2[m2]) / b2[m2])) * 100) if m2.any() else float('nan')

    mse_avg = (mse1 + mse2) / 2.0
    if (not np.isnan(mape1)) and (not np.isnan(mape2)):
        mape_avg = (mape1 + mape2) / 2.0
    else:
        mape_avg = mape1 if not np.isnan(mape1) else mape2

    print(f"[Workload {i}] {node1_name}: MSE={mse1:.6f}, MAPE={mape1:.3f}% | "
          f"{node2_name}: MSE={mse2:.6f}, MAPE={mape2:.3f}% | "
          f"AVG MSE={mse_avg:.6f}, AVG MAPE={mape_avg:.3f}%")

    # ---------- plot (2 subplots; tasks only) ----------
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    ax1, ax2 = axes

    for k in range(50):
        s = df1.iloc[:, k]
        if (s != 0).any():
            ax1.plot(s.index, s.values, label=f"Task{str(k+1).zfill(2)}", linewidth=1, alpha=0.9)
    ax1.set_title(f"Workload {i} — {node1_name} (True Blockwise NNLS)")
    ax1.set_ylabel("CPU Util (%)")
    h1, l1 = ax1.get_legend_handles_labels()
    if l1:
        ax1.legend(h1, l1, loc="upper right", ncol=2, fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.3)

    for k in range(50):
        s = df2.iloc[:, k]
        if (s != 0).any():
            ax2.plot(s.index, s.values, label=f"Task{str(k+1).zfill(2)}", linewidth=1, alpha=0.9)
    ax2.set_title(f"Workload {i} — {node2_name} (True Blockwise NNLS)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("CPU Util (%)")
    h2, l2 = ax2.get_legend_handles_labels()
    if l2:
        ax2.legend(h2, l2, loc="upper right", ncol=2, fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"bnnls{i}.png", dpi=300)
    plt.close()
