# ---------------------- imports ----------------------
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import nnls

# ---------------------- load data ----------------------
with open("all_workloads_ic2.json", "r") as f:
    workloads = json.load(f)
with open("all_system_loads_ic2.json", "r") as f:
    sysloads = json.load(f)

# ---------------------- run first 25 workloads ----------------------
for i in range(25):
    workload = workloads[i]
    tasklist = workload["tasklist"]
    sys_entry = sysloads[i]
    node_list = sys_entry["node_list"]

    # ---------- get two node names ----------
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

    # ---------- pull CPU time series ----------
    t_n1 = None; u_n1 = None; t_n2 = None; u_n2 = None
    for nd in node_list:
        if nd["node_name"] == node1_name:
            arr = nd["metrics"]["cpu_util"]
            t_n1 = np.array([float(x[0]) for x in arr], dtype=float)
            u_n1 = np.array([float(x[1]) for x in arr], dtype=float)
        elif nd["node_name"] == node2_name:
            arr = nd["metrics"]["cpu_util"]
            t_n2 = np.array([float(x[0]) for x in arr], dtype=float)
            u_n2 = np.array([float(x[1]) for x in arr], dtype=float)
    if t_n1 is None or t_n2 is None:
        print(f"[Workload {i}] Missing cpu series for one node. Skipping.")
        continue

    idx1 = pd.to_datetime(t_n1, unit='s')
    idx2 = pd.to_datetime(t_n2, unit='s')
    cpu1 = pd.Series(u_n1, index=idx1)
    cpu2 = pd.Series(u_n2, index=idx2)

    # ---------- collect up to 50 tasks per node (start,end) ----------
    tasks_node1 = []   # (task_idx, start, end)
    tasks_node2 = []
    c1 = 0; c2 = 0
    for t in tasklist:
        s = float(t["start_time"]); e = float(t["finish_time"])
        for nd in t["nodes"]:
            n = nd["node_name"]
            if n == node1_name and c1 < 50:
                tasks_node1.append((c1, s, e)); c1 += 1
            elif n == node2_name and c2 < 50:
                tasks_node2.append((c2, s, e)); c2 += 1
        if c1 >= 50 and c2 >= 50:
            break

    # ---------- build A (binary activity) and b ----------
    # Node 1
    T1 = len(cpu1.index); n1 = len(tasks_node1)
    A1 = np.zeros((T1, n1), dtype=float)
    for (j, s, e) in tasks_node1:
        m = (cpu1.index.view('i8')/1e9 >= s) & (cpu1.index.view('i8')/1e9 < e)
        A1[np.nonzero(m)[0], j] = 1.0
    b1 = cpu1.values.astype(float)

    # Node 2
    T2 = len(cpu2.index); n2 = len(tasks_node2)
    A2 = np.zeros((T2, n2), dtype=float)
    for (j, s, e) in tasks_node2:
        m = (cpu2.index.view('i8')/1e9 >= s) & (cpu2.index.view('i8')/1e9 < e)
        A2[np.nonzero(m)[0], j] = 1.0
    b2 = cpu2.values.astype(float)

    # ---------- blockwise NNLS using scipy.optimize.nnls ----------
    # settings
    max_outer = 30        # passes over blocks
    block_size = 10       # columns per block

    # Node 1 solve
    if n1 > 0:
        x1 = np.zeros(n1, dtype=float)
        y1 = A1 @ x1
        blocks1 = [np.arange(s, min(s+block_size, n1)) for s in range(0, n1, block_size)]
        for _ in range(max_outer):
            for blk in blocks1:
                Ablk = A1[:, blk]          # (T1 x nb)
                xblk_old = x1[blk].copy()
                y1 -= Ablk @ xblk_old      # remove old block contribution
                r = b1 - y1                 # residual for this block
                xblk, _ = nnls(Ablk, r)     # true NNLS on the block
                y1 += Ablk @ xblk           # add new block contribution
                x1[blk] = xblk
        recon1 = y1
    else:
        x1 = np.zeros(0, dtype=float)
        recon1 = np.zeros(T1, dtype=float)

    # Node 2 solve
    if n2 > 0:
        x2 = np.zeros(n2, dtype=float)
        y2 = A2 @ x2
        blocks2 = [np.arange(s, min(s+block_size, n2)) for s in range(0, n2, block_size)]
        for _ in range(max_outer):
            for blk in blocks2:
                Ablk = A2[:, blk]
                xblk_old = x2[blk].copy()
                y2 -= Ablk @ xblk_old
                r = b2 - y2
                xblk, _ = nnls(Ablk, r)
                y2 += Ablk @ xblk
                x2[blk] = xblk
        recon2 = y2
    else:
        x2 = np.zeros(0, dtype=float)
        recon2 = np.zeros(T2, dtype=float)

    # ---------- per-task time series (A * diag(x)) ----------
    cols1 = [f"Task{str(k+1).zfill(2)}" for k in range(50)]
    cols2 = [f"Task{str(k+1).zfill(2)}" for k in range(50)]

    df1 = pd.DataFrame(0.0, index=cpu1.index, columns=cols1)
    for idx in range(len(x1)):
        if x1[idx] != 0.0:
            df1.iloc[:, idx] = A1[:, idx] * x1[idx]

    df2 = pd.DataFrame(0.0, index=cpu2.index, columns=cols2)
    for idx in range(len(x2)):
        if x2[idx] != 0.0:
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
    ax1.set_title(f"Workload {i} — {node1_name} (Blockwise NNLS via scipy.optimize.nnls)")
    ax1.set_ylabel("CPU Util (%)")
    h1, l1 = ax1.get_legend_handles_labels()
    if l1:
        ax1.legend(h1, l1, loc="upper right", ncol=2, fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.3)

    for k in range(50):
        s = df2.iloc[:, k]
        if (s != 0).any():
            ax2.plot(s.index, s.values, label=f"Task{str(k+1).zfill(2)}", linewidth=1, alpha=0.9)
    ax2.set_title(f"Workload {i} — {node2_name} (Blockwise NNLS via scipy.optimize.nnls)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("CPU Util (%)")
    h2, l2 = ax2.get_legend_handles_labels()
    if l2:
        ax2.legend(h2, l2, loc="upper right", ncol=2, fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"nnls{i}.png", dpi=300)
    plt.close()

