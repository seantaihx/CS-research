
from algorithm_blockwise_nonnegative_least_squares import pertask_utilization_NNLS
from algorithm_greedy_mean_only import pertask_utilization_greedy
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


contribs_nnls = pertask_utilization_NNLS()
contribs_greedy = pertask_utilization_greedy()

OUT_DIR = "plots_ic2_memory_per_node"
os.makedirs(OUT_DIR, exist_ok=True)
FIGSIZE = (21,18)
USE_RELATIVE_TIME = True

cmap = plt.colormaps.get_cmap("tab20")

common_keys = sorted(set(contribs_nnls) & set(contribs_greedy))
#print("COMMON (workload,node) count:", len(common_keys))


metrics_rows = []
# columns: wname,node,task_id,n_points,mean_avg,mse,mae,norm_mse,norm_mae
# mean_avg = mean((a+b)/2)
big_counter = 0
all_mse = 0.0
all_mae = 0.0
for (wname, node) in common_keys:
    nnls_list = contribs_nnls[(wname, node)]
    greedy_list = contribs_greedy[(wname, node)]

    # build tid -> {t:u}
    nnls_map = {int(r["task_id"]): {float(t): float(u) for (t, u) in r["Util"]} for r in nnls_list}
    greedy_map = {int(r["task_id"]): {float(t): float(u) for (t, u) in r["Util"]} for r in greedy_list}

    # all tasks active in either algorithm
    all_tids = sorted(set(nnls_map) | set(greedy_map))
    if not all_tids:
        continue

    fig = plt.figure(figsize=FIGSIZE)
    ax = plt.gca()

    total_mse = 0.0
    total_mae = 0.0
    plotted = 0
    counter = 0
    for i, tid in enumerate(all_tids):
        a = nnls_map.get(tid, {})
        b = greedy_map.get(tid, {})

        # aligned union timestamps
        xs = np.array(sorted(set(a) | set(b)), dtype=float)
        if xs.size < 2:
            continue

        y_a = np.array([a.get(x, 0.0) for x in xs], dtype=float)   # NNLS
        y_b = np.array([b.get(x, 0.0) for x in xs], dtype=float)   # Greedy

        # define "active task": has any non-trivial util in either algo
        if (y_a.max() <= 0) and (y_b.max() <= 0):
            continue

        xs_plot = xs

        color = cmap(i % cmap.N)

        # plot lines
        ax.plot(xs_plot, y_a, color=color, linewidth=1.2, linestyle="-", label=f"Task {tid}")
        ax.plot(xs_plot, y_b, color=color, linewidth=1.2, linestyle="--")

        # fill only makes sense when both exist (but even if one is all zeros, it's fine)
        ax.fill_between(xs_plot, y_a, y_b, color=color, alpha=0.15)

        plotted += 1

        # -----------------------
        # Metrics (per task)
        # -----------------------

        mean_avg = float(np.mean((y_a + y_b) / 2.0))
        mse = float(np.mean((y_a - y_b) ** 2))
        mae = float(np.mean(np.abs(y_a - y_b)))

        if mean_avg > 0:
            norm_mse = mse / mean_avg
            norm_mae = mae / mean_avg
        else:
            norm_mse = np.nan
            norm_mae = np.nan

        total_mse += norm_mse
        total_mae += norm_mae

        #print("mse:", norm_mse)
        #print("mae:", norm_mae)

        metrics_rows.append([
            wname, node, tid, int(xs.size),
            mean_avg, mse, mae, norm_mse, norm_mae
        ])
        counter += 1
    big_counter += 1
    print("total mse:", total_mse/counter)
    print("total mae:", total_mae/counter)
    print("counter:", counter)
    
    if plotted == 0:
        plt.close(fig)
        continue
    all_mse += total_mse/counter
    all_mae += total_mae/counter
    ax.set_title(f"{wname} | {node} | {plotted} active tasks (NNLS solid vs Greedy dashed)")
    ax.set_xlabel("timestamp (relative)" if USE_RELATIVE_TIME else "timestamp")
    ax.set_ylabel("utilization")

    task_handles, task_labels = ax.get_legend_handles_labels()

    task_legend = ax.legend(
        task_handles,
        task_labels,
        title="Task ID (color)",
        loc="upper left",
        fontsize=8,
        title_fontsize=9
    )

    ax.add_artist(task_legend)  # 🔒 lock task legend

    ax.legend(
        handles = [
            Line2D([0], [0], color="black", linestyle="-", label="NNLS"),
            Line2D([0], [0], color="black", linestyle="--", label="Greedy"),
        ],
        title="Algorithm",
        loc="upper right",
        fontsize=8,
        title_fontsize=9
    )

    safe_w = "".join(c if c.isalnum() else "_" for c in str(wname))
    safe_n = "".join(c if c.isalnum() else "_" for c in str(node))
    out_path = os.path.join(OUT_DIR, f"{safe_w}_{safe_n}_compare.png")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

print("Plots saved to:", OUT_DIR)
print("Overall avg mse:", all_mse/big_counter)
print("Overall avg mae:", all_mae/big_counter)