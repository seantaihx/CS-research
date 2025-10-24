# ---------------------- imports ----------------------
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- load files ----------------------
with open("all_workloads_ic2.json", "r") as f:
    workloads = json.load(f)
with open("all_system_loads_ic2.json", "r") as f:
    sysloads = json.load(f)

# ---------------------- loop through workloads ----------------------
for WIDX in range(25):
    tasks = workloads[WIDX]["tasklist"]
    node_list = sysloads[WIDX]["node_list"]

    # ---------------- detect two node names ----------------
    node_names = []
    seen = set()
    for task in tasks:
        for n in task["nodes"]:
            nm = n["node_name"]
            if nm not in seen:
                seen.add(nm)
                node_names.append(nm)
            if len(node_names) == 2:
                break
        if len(node_names) == 2:
            break

    node1_name, node2_name = node_names[0], node_names[1]

    # ---------------- create figure for both nodes ----------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    ax1, ax2 = axes

    print(f"\n========== Workload {WIDX} ==========")
    for ax, node_name in zip((ax1, ax2), (node1_name, node2_name)):

        # CPU util series
        nd = next(n for n in node_list if n["node_name"] == node_name)
        pairs = nd["metrics"]["cpu_util"]
        cpu_times = pd.to_datetime([float(ts) for ts, _ in pairs], unit="s")
        cpu_vals  = [float(val) for _, val in pairs]
        cpu = pd.Series(cpu_vals, index=cpu_times, name=f"{node_name}_cpu").sort_index()

        # Jobs for this node
        jobs = []
        for task in tasks:
            if any(n.get("node_name") == node_name for n in task.get("nodes", [])):
                jobs.append({
                    "id": int(task.get("job_id", len(jobs) + 1)),
                    "start": pd.to_datetime(float(task["start_time"]), unit="s"),
                    "end":   pd.to_datetime(float(task["finish_time"]), unit="s"),
                })
        jobs.sort(key=lambda j: j["id"])

        # Master time grid
        grid = sorted(set(cpu.index) | {j["start"] for j in jobs} | {j["end"] for j in jobs})
        grid = [t for t in grid if (t >= cpu.index.min()) and (t <= cpu.index.max())]
        cpu_on_grid = cpu.reindex(grid).ffill() 

        # Intervals and observed CPU at left edges
        intervals = list(zip(grid[:-1], grid[1:]))
        left_times = pd.Index([t0 for t0, _ in intervals])
        y_t = cpu_on_grid.iloc[:-1].to_numpy(dtype=float)
        T = len(intervals)
        J = len(jobs)

        # Active mask (A[t,j] = 1 if job covers interval)
        A = np.zeros((T, J), dtype=bool)
        for t_idx, (t0, t1) in enumerate(intervals):
            for j_idx, job in enumerate(jobs):
                if (job["start"] <= t0) and (t1 <= job["end"]):
                    A[t_idx, j_idx] = True

        # Greedy equal split
        k = A.sum(axis=1)
        share = np.zeros(T, dtype=float)
        nz = k > 0
        share[nz] = y_t[nz] / k[nz]

        # Constructed total CPU (sum of all per-job shares)
        y_hat = (A * share[:, None]).sum(axis=1)

        # MSE & MAPE
        MSE = np.sum((y_hat - y_t) ** 2) / len(y_t)
        mask = y_t != 0
        MAPE = np.sum(np.abs(y_hat[mask] - y_t[mask]) / y_t[mask] * 100) / len(y_t)
        print(f"[{node_name}]  MSE = {MSE:.6f},  MAPE = {MAPE:.6f}%")

        # 9️⃣ Plot per-job utilization (up to 50)
'''        max_jobs = 50
        handles_labels = []
        for j_idx, job in enumerate(jobs[:max_jobs]):
            vals = np.where(A[:, j_idx], share, 0.0)
            s = pd.Series(vals, index=left_times)
            h, = ax.plot(s.index, s.values, drawstyle="steps-post",
                         linewidth=1.0, alpha=0.75, label=f"job {job['id']}")
            handles_labels.append((job["id"], h))

        # 🔟 Sort legend by job id
        handles_labels.sort(key=lambda x: x[0])
        handles_sorted = [h for _, h in handles_labels]
        labels_sorted = [f"job {jid}" for jid, _ in handles_labels]
        ax.legend(handles_sorted, labels_sorted, fontsize=8, ncol=2, loc="upper right")

        # 🔢 Styling
        ax.set_title(f"Node: {node_name} — Greedy Mean per-job CPU (jobs 1–50)")
        ax.set_ylabel("CPU Utilization")
        ax.grid(True, linestyle="--", alpha=0.3)

    ax2.set_xlabel("Time")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"greedy{WIDX}.png", dpi=300)
'''