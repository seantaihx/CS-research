import numpy as np
import matplotlib.pyplot as plt
import json
from algorithm_blockwise_nonnegative_least_squares import nnls_main
from algorithm_greedy_mean_only import gm_main
from gpu_imbalance_correlation import gpu_imbalance

def _to_1d_array(x):
    """Convert list/Series/np array -> 1D float array, dropping NaNs."""
    arr = np.asarray(x, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def _cdf_xy(values):
    """Return x (sorted) and y (CDF) for a 1D array."""
    x = np.sort(_to_1d_array(values))
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _percentile(values, p):
    """p in [0,100]."""
    v = _to_1d_array(values)
    if v.size == 0:
        return np.nan
    return float(np.percentile(v, p))

def plot_cdf_gpu(gpu_ic2, gpu_polaris,*,
                 p_tail=95,
                 title_prefix="GPU Utilization CDF across workloads",
                 xlabel="GPU Utilization (%)",
                 save_path=None,
                 show=True):
    
    def _extract(gpu_utils):
        if isinstance(gpu_utils, dict):
            return list(gpu_utils.values())
        return gpu_utils
    
    gpu_utils_ic2 = _extract(gpu_ic2)
    gpu_utils_polaris = _extract(gpu_polaris)
    
    # --- CPU CDF ---
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fig.suptitle(
        "GPU CDF across workloads",
        fontsize=20,
        fontweight="bold"
    )

    x1, y1 = _cdf_xy(gpu_utils_ic2)
    x2, y2 = _cdf_xy(gpu_utils_polaris)

    axes[0].plot(x1, y1, label="IC2")
    axes[1].plot(x2, y2, label="Polaris")

        # percentiles
    for label, values in [("IC2", gpu_utils_ic2), ("Polaris", gpu_utils_polaris)]:
        p50 = _percentile(values, 50)
        pt = _percentile(values, p_tail)
        if np.isfinite(p50):
            if label == "IC2":
                axes[0].axvline(p50, linestyle="--", linewidth=1)
            else:
                axes[1].axvline(p50, linestyle="--", linewidth=1)
        if np.isfinite(pt):
            if label == "IC2":
                axes[0].axvline(pt, linestyle=":", linewidth=1)
            else:
                axes[1].axvline(pt, linestyle=":", linewidth=1)
    for i in range(2):
        axes[i].set_title(f"{title_prefix}")
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel("CDF")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()


    # Print numbers you can cite in text
    gpu_p50_ic2, gpu_p95_ic2 = _percentile(gpu_utils_ic2, 50), _percentile(gpu_utils_ic2, p_tail)
    gpu_p50_polaris, gpu_p95_polaris = _percentile(gpu_utils_polaris, 50), _percentile(gpu_utils_polaris, p_tail)
    print("GPU Utilization percentiles")
    print(f"  IC2:       p50={gpu_p50_ic2:.6g}, p{p_tail}={gpu_p95_ic2:.6g}")
    print(f"  Polaris: p50={gpu_p50_polaris:.6g}, p{p_tail}={gpu_p95_polaris:.6g}")
    if np.isfinite(gpu_p95_ic2) and gpu_p95_ic2 > 0:
        print(f"  Tail ratio (ic2/polaris) at p{p_tail}: {gpu_p95_ic2 / gpu_p95_polaris:.3g}")

    # Save if requested
    if save_path:
        # save_path can be a prefix; we save two files
        fig.savefig(f"{save_path}_{type}_cdf.png", dpi=300, bbox_inches="tight")


    if show:
        plt.show()

    return {
        "ic2": {"ic2_p50": gpu_p50_ic2, f"ic2_p{p_tail}": gpu_p95_ic2},
        "polaris": {"polaris_p50": gpu_p50_polaris, f"polaris_p{p_tail}": gpu_p95_polaris,}
    }
        

def plot_cdf_with_percentiles(
    cpu_mae_nnls,
    cpu_mae_greedy,
    mem_mae_nnls,
    mem_mae_greedy,
    type,
    *,
    cpu_mae_nnls2 = None,
    cpu_mae_greedy2 = None,
    mem_mae_nnls2 = None,
    mem_mae_greedy2 = None,
    label_1 = "IC2",
    label_2 = "Polaris",
    p_tail=95,
    title_prefix="MAE CDF across workloads",
    xlabel_cpu="CPU MAE",
    xlabel_mem="Memory MAE",
    save_path=None,
    show=True,
):
    """
    Plots two CDF figures (CPU, Memory) comparing NNLS vs Greedy Mean,
    with vertical lines at median (p50) and p_tail (default p95).

    Inputs can be:
      - list/np.array of per-workload MAE values, OR
      - dict {workload_id: mae} (we'll plot the values)

    Example:
      cpu_nnls = {"w1": 0.12, "w2": 0.08, ...}
      cpu_gm   = {"w1": 0.20, "w2": 0.11, ...}
    """

    def _extract(v):
        if isinstance(v, dict):
            return list(v.values())
        return v

    type = type.strip().lower()



    if type != "both":
        cpu_nnls = _extract(cpu_mae_nnls)
        cpu_gm = _extract(cpu_mae_greedy)
        mem_nnls = _extract(mem_mae_nnls)
        mem_gm = _extract(mem_mae_greedy)

        # --- CPU CDF ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        fig.suptitle(
            "MAE CDF across workloads",
            fontsize=20,
            fontweight="bold"
        )

        x1, y1 = _cdf_xy(cpu_nnls)
        x2, y2 = _cdf_xy(cpu_gm)

        axes[0].plot(x1, y1, label="NNLS")
        axes[0].plot(x2, y2, label="Greedy Mean")

        # percentiles
        for label, values in [("NNLS", cpu_nnls), ("Greedy Mean", cpu_gm)]:
            p50 = _percentile(values, 50)
            pt = _percentile(values, p_tail)
            if np.isfinite(p50):
                axes[0].axvline(p50, linestyle="--", linewidth=1)
            if np.isfinite(pt):
                axes[0].axvline(pt, linestyle=":", linewidth=1)

        axes[0].set_title(f"{title_prefix}: CPU")
        axes[0].set_xlabel(xlabel_cpu)
        axes[0].set_ylabel("CDF")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Print numbers you can cite in text
        cpu_p50_nnls, cpu_p95_nnls = _percentile(cpu_nnls, 50), _percentile(cpu_nnls, p_tail)
        cpu_p50_gm, cpu_p95_gm = _percentile(cpu_gm, 50), _percentile(cpu_gm, p_tail)
        print("CPU MAE percentiles")
        print(f"  NNLS:       p50={cpu_p50_nnls:.6g}, p{p_tail}={cpu_p95_nnls:.6g}")
        print(f"  GreedyMean: p50={cpu_p50_gm:.6g}, p{p_tail}={cpu_p95_gm:.6g}")
        if np.isfinite(cpu_p95_nnls) and cpu_p95_nnls > 0:
            print(f"  Tail ratio (GreedyMean/NNLS) at p{p_tail}: {cpu_p95_gm / cpu_p95_nnls:.3g}")

        # --- Memory CDF ---
        x1, y1 = _cdf_xy(mem_nnls)
        x2, y2 = _cdf_xy(mem_gm)

        axes[1].plot(x1, y1, label="NNLS")
        axes[1].plot(x2, y2, label="Greedy Mean")

        for label, values in [("NNLS", mem_nnls), ("Greedy Mean", mem_gm)]:
            p50 = _percentile(values, 50)
            pt = _percentile(values, p_tail)
            if np.isfinite(p50):
                axes[1].axvline(p50, linestyle="--", linewidth=1)
            if np.isfinite(pt):
                axes[1].axvline(pt, linestyle=":", linewidth=1)

        axes[1].set_title(f"{title_prefix}: Memory")
        axes[1].set_xlabel(xlabel_mem)
        axes[1].set_ylabel("CDF")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        mem_p50_nnls, mem_p95_nnls = _percentile(mem_nnls, 50), _percentile(mem_nnls, p_tail)
        mem_p50_gm, mem_p95_gm = _percentile(mem_gm, 50), _percentile(mem_gm, p_tail)
        print("\nMemory MAE percentiles")
        print(f"  NNLS:       p50={mem_p50_nnls:.6g}, p{p_tail}={mem_p95_nnls:.6g}")
        print(f"  GreedyMean: p50={mem_p50_gm:.6g}, p{p_tail}={mem_p95_gm:.6g}")
        if np.isfinite(mem_p95_nnls) and mem_p95_nnls > 0:
            print(f"  Tail ratio (GreedyMean/NNLS) at p{p_tail}: {mem_p95_gm / mem_p95_nnls:.3g}")

        # Save if requested
        if save_path:
            # save_path can be a prefix; we save two files
            fig.savefig(f"{save_path}_{type}_cdf.png", dpi=300, bbox_inches="tight")


        if show:
            plt.show()

        return {
            "cpu": {"nnls_p50": cpu_p50_nnls, f"nnls_p{p_tail}": cpu_p95_nnls,
                    "gm_p50": cpu_p50_gm, f"gm_p{p_tail}": cpu_p95_gm},
            "mem": {"nnls_p50": mem_p50_nnls, f"nnls_p{p_tail}": mem_p95_nnls,
                    "gm_p50": mem_p50_gm, f"gm_p{p_tail}": mem_p95_gm},
        }

    elif type == "both":
        cpu1_nnls = _extract(cpu_mae_nnls)
        cpu1_gm = _extract(cpu_mae_greedy)
        mem1_nnls = _extract(mem_mae_nnls)
        mem1_gm = _extract(mem_mae_greedy)
        cpu2_nnls = _extract(cpu_mae_nnls2)
        cpu2_gm = _extract(cpu_mae_greedy2)
        mem2_nnls = _extract(mem_mae_nnls2)
        mem2_gm = _extract(mem_mae_greedy2)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        '''
        fig.suptitle(
            "MAE CDF across workloads",
            fontsize=20,
            fontweight="bold"
        )
        '''

        x1, y1 = _cdf_xy(cpu1_nnls)
        x2, y2 = _cdf_xy(cpu1_gm)
        axes[0, 0].plot(x1, y1, label="NNLS")
        axes[0, 0].plot(x2, y2, label="Greedy Mean")
        for label, values in [("NNLS", cpu1_nnls), ("Greedy Mean", cpu1_gm)]:
            p50 = _percentile(values, 50)
            pt  = _percentile(values, p_tail)
            if np.isfinite(p50):
                axes[0, 0].axvline(p50, linestyle="--", linewidth=1)
            if np.isfinite(pt):
                axes[0, 0].axvline(pt, linestyle=":", linewidth=1)
        axes[0, 0].set_title(f"{title_prefix}: {label_1} CPU")
        axes[0, 0].set_xlabel(xlabel_cpu)
        axes[0, 0].set_ylabel("CDF")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        cpu1_p50_nnls, cpu1_p95_nnls = _percentile(cpu1_nnls, 50), _percentile(cpu1_nnls, p_tail)
        cpu1_p50_gm,   cpu1_p95_gm   = _percentile(cpu1_gm, 50),   _percentile(cpu1_gm, p_tail)
        print(f"\n{label_1} CPU MAE percentiles")
        print(f"  NNLS:       p50={cpu1_p50_nnls:.6g}, p{p_tail}={cpu1_p95_nnls:.6g}")
        print(f"  GreedyMean: p50={cpu1_p50_gm:.6g}, p{p_tail}={cpu1_p95_gm:.6g}")
        if np.isfinite(cpu1_p95_nnls) and cpu1_p95_nnls > 0 and np.isfinite(cpu1_p95_gm):
            print(f"  Tail ratio (GreedyMean/NNLS) at p{p_tail}: {cpu1_p95_gm / cpu1_p95_nnls:.3g}")

        # ---- (0,1) label_1 Memory ----
        x1, y1 = _cdf_xy(mem1_nnls)
        x2, y2 = _cdf_xy(mem1_gm)
        axes[0, 1].plot(x1, y1, label="NNLS")
        axes[0, 1].plot(x2, y2, label="Greedy Mean")
        for label, values in [("NNLS", mem1_nnls), ("Greedy Mean", mem1_gm)]:
            p50 = _percentile(values, 50)
            pt  = _percentile(values, p_tail)
            if np.isfinite(p50):
                axes[0, 1].axvline(p50, linestyle="--", linewidth=1)
            if np.isfinite(pt):
                axes[0, 1].axvline(pt, linestyle=":", linewidth=1)
        axes[0, 1].set_title(f"{title_prefix}: {label_1} Memory")
        axes[0, 1].set_xlabel(xlabel_mem)
        axes[0, 1].set_ylabel("CDF")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        mem1_p50_nnls, mem1_p95_nnls = _percentile(mem1_nnls, 50), _percentile(mem1_nnls, p_tail)
        mem1_p50_gm,   mem1_p95_gm   = _percentile(mem1_gm, 50),   _percentile(mem1_gm, p_tail)
        print(f"\n{label_1} Memory MAE percentiles")
        print(f"  NNLS:       p50={mem1_p50_nnls:.6g}, p{p_tail}={mem1_p95_nnls:.6g}")
        print(f"  GreedyMean: p50={mem1_p50_gm:.6g}, p{p_tail}={mem1_p95_gm:.6g}")
        if np.isfinite(mem1_p95_nnls) and mem1_p95_nnls > 0 and np.isfinite(mem1_p95_gm):
            print(f"  Tail ratio (GreedyMean/NNLS) at p{p_tail}: {mem1_p95_gm / mem1_p95_nnls:.3g}")

        # ---- (1,0) label_2 CPU ----
        x1, y1 = _cdf_xy(cpu2_nnls)
        x2, y2 = _cdf_xy(cpu2_gm)
        axes[1, 0].plot(x1, y1, label="NNLS")
        axes[1, 0].plot(x2, y2, label="Greedy Mean")
        for label, values in [("NNLS", cpu2_nnls), ("Greedy Mean", cpu2_gm)]:
            p50 = _percentile(values, 50)
            pt  = _percentile(values, p_tail)
            if np.isfinite(p50):
                axes[1, 0].axvline(p50, linestyle="--", linewidth=1)
            if np.isfinite(pt):
                axes[1, 0].axvline(pt, linestyle=":", linewidth=1)
        axes[1, 0].set_title(f"{title_prefix}: {label_2} CPU")
        axes[1, 0].set_xlabel(xlabel_cpu)
        axes[1, 0].set_ylabel("CDF")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        cpu2_p50_nnls, cpu2_p95_nnls = _percentile(cpu2_nnls, 50), _percentile(cpu2_nnls, p_tail)
        cpu2_p50_gm,   cpu2_p95_gm   = _percentile(cpu2_gm, 50),   _percentile(cpu2_gm, p_tail)
        print(f"\n{label_2} CPU MAE percentiles")
        print(f"  NNLS:       p50={cpu2_p50_nnls:.6g}, p{p_tail}={cpu2_p95_nnls:.6g}")
        print(f"  GreedyMean: p50={cpu2_p50_gm:.6g}, p{p_tail}={cpu2_p95_gm:.6g}")
        if np.isfinite(cpu2_p95_nnls) and cpu2_p95_nnls > 0 and np.isfinite(cpu2_p95_gm):
            print(f"  Tail ratio (GreedyMean/NNLS) at p{p_tail}: {cpu2_p95_gm / cpu2_p95_nnls:.3g}")

        # ---- (1,1) label_2 Memory ----
        x1, y1 = _cdf_xy(mem2_nnls)
        x2, y2 = _cdf_xy(mem2_gm)
        axes[1, 1].plot(x1, y1, label="NNLS")
        axes[1, 1].plot(x2, y2, label="Greedy Mean")
        for values in [mem2_nnls, mem2_gm]:
            p50 = _percentile(values, 50)
            pt  = _percentile(values, p_tail)
            if np.isfinite(p50):
                axes[1, 1].axvline(p50, linestyle="--", linewidth=1)
            if np.isfinite(pt):
                axes[1, 1].axvline(pt, linestyle=":", linewidth=1)
        axes[1, 1].set_title(f"{title_prefix}: {label_2} Memory")
        axes[1, 1].set_xlabel(xlabel_mem)
        axes[1, 1].set_ylabel("CDF")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        mem2_p50_nnls, mem2_p95_nnls = _percentile(mem2_nnls, 50), _percentile(mem2_nnls, p_tail)
        mem2_p50_gm,   mem2_p95_gm   = _percentile(mem2_gm, 50),   _percentile(mem2_gm, p_tail)
        print(f"\n{label_2} Memory MAE percentiles")
        print(f"  NNLS:       p50={mem2_p50_nnls:.6g}, p{p_tail}={mem2_p95_nnls:.6g}")
        print(f"  GreedyMean: p50={mem2_p50_gm:.6g}, p{p_tail}={mem2_p95_gm:.6g}")
        if np.isfinite(mem2_p95_nnls) and mem2_p95_nnls > 0 and np.isfinite(mem2_p95_gm):
            print(f"  Tail ratio (GreedyMean/NNLS) at p{p_tail}: {mem2_p95_gm / mem2_p95_nnls:.3g}")

    
        if save_path:
            fig.savefig(f"{save_path}_all_cdf.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return {
            label_1: {
                "cpu": {"nnls_p50": cpu1_p50_nnls, f"nnls_p{p_tail}": cpu1_p95_nnls,
                        "gm_p50": cpu1_p50_gm, f"gm_p{p_tail}": cpu1_p95_gm},
                "mem": {"nnls_p50": mem1_p50_nnls, f"nnls_p{p_tail}": mem1_p95_nnls,
                        "gm_p50": mem1_p50_gm, f"gm_p{p_tail}": mem1_p95_gm},
            },
            label_2: {
                "cpu": {"nnls_p50": cpu2_p50_nnls, f"nnls_p{p_tail}": cpu2_p95_nnls,
                        "gm_p50": cpu2_p50_gm, f"gm_p{p_tail}": cpu2_p95_gm},
                "mem": {"nnls_p50": mem2_p50_nnls, f"nnls_p{p_tail}": mem2_p95_nnls,
                        "gm_p50": mem2_p50_gm, f"gm_p{p_tail}": mem2_p95_gm},
            }
        }

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    
    prompt = input("What to plot? (ic2/polaris/both): ").strip().lower()
    if prompt == "ic2":
        workload_file = input("Enter IC2 workload file: ")
        system_file = input("Enter IC2 system load file: ")
        with open(workload_file, "r") as f1:
            workloads = json.load(f1)
        with open(system_file, "r") as f2:
            system_loads = json.load(f2)
        cpu_nnls, cpu_nnls_node = nnls_main(workloads, system_loads, utilization="cpu")
        mem_nnls, mem_nnls_node = nnls_main(workloads, system_loads, utilization="memory")
        cpu_gm, cpu_gm_node = gm_main(workloads, system_loads, utilization="cpu")
        mem_gm, mem_gm_node = gm_main(workloads, system_loads, utilization="memory")
        plot_cdf_with_percentiles(cpu_nnls_node, cpu_gm_node, mem_nnls_node, mem_gm_node, type="ic2", p_tail=95, save_path=True, show=True)
       

    elif prompt == "polaris":
        workload_file = input("Enter Polaris workload file: ")
        system_file = input("Enter Polaris system load file: ")
        with open(workload_file, "r") as f1:
            workloads = json.load(f1)
        with open(system_file, "r") as f2:
            system_loads = json.load(f2)
        cpu_nnls, cpu_nnls_node = nnls_main(workloads, system_loads, utilization="cpu")
        mem_nnls, mem_nnls_node = nnls_main(workloads, system_loads, utilization="memory")
        cpu_gm, cpu_gm_node = gm_main(workloads, system_loads, utilization="cpu")
        mem_gm, mem_gm_node = gm_main(workloads, system_loads, utilization="memory")
        plot_cdf_with_percentiles(cpu_nnls_node, cpu_gm_node, mem_nnls_node, mem_gm_node, type="polaris", p_tail=95, save_path=True, show=True)


    elif prompt == "both":
        workload_file_ic2 = input("Enter IC2 workload file: ")
        system_file_ic2 = input("Enter IC2 system load file: ")
        workload_file_polaris = input("Enter Polaris workload file: ")
        system_file_polaris = input("Enter Polaris system load file: ")
        with open(workload_file_ic2, "r") as f1:
            workloads_ic2 = json.load(f1)
        with open(system_file_ic2, "r") as f2:
            system_loads_ic2 = json.load(f2)
        with open(workload_file_polaris, "r") as f3:
            workloads_polaris = json.load(f3)
        with open(system_file_polaris, "r") as f4:
            system_loads_polaris = json.load(f4)
        cpu_nnls_ic2, cpu_nnls_node_ic2 = nnls_main(workloads_ic2, system_loads_ic2, utilization="cpu")
        mem_nnls_ic2, mem_nnls_node_ic2 = nnls_main(workloads_ic2, system_loads_ic2, utilization="memory")
        cpu_gm_ic2, cpu_gm_node_ic2 = gm_main(workloads_ic2, system_loads_ic2, utilization="cpu")
        mem_gm_ic2, mem_gm_node_ic2 = gm_main(workloads_ic2, system_loads_ic2, utilization="memory")
        cpu_nnls_polaris, cpu_nnls_node_polaris = nnls_main(workloads_polaris, system_loads_polaris, utilization="cpu")
        mem_nnls_polaris, mem_nnls_node_polaris = nnls_main(workloads_polaris, system_loads_polaris, utilization="memory")
        cpu_gm_polaris, cpu_gm_node_polaris = gm_main(workloads_polaris, system_loads_polaris, utilization="cpu")
        mem_gm_polaris, mem_gm_node_polaris = gm_main(workloads_polaris, system_loads_polaris, utilization="memory")
        plot_cdf_with_percentiles(cpu_nnls_node_ic2, cpu_gm_node_ic2, mem_nnls_node_ic2, mem_gm_node_ic2, type="both", 
                                  cpu_mae_nnls2 = cpu_nnls_node_polaris, cpu_mae_greedy2 = cpu_gm_node_polaris, mem_mae_nnls2 = mem_nnls_node_polaris, mem_mae_greedy2 = mem_gm_node_polaris,
                                  p_tail=95, save_path=True, show=True)
        
    elif prompt == "gpu":
        workload_file_ic2 = "all_workloads_ic2.json"
        workload_file_polaris = "all_workloads_polaris.json"
        #workload_file_ic2 = input("Enter IC2 workload file: ")
        #workload_file_polaris = input("Enter Polaris workload file: ")
        with open(workload_file_ic2, "r") as f1:
            workloads_ic2 = json.load(f1)
        with open(workload_file_polaris, "r") as f2:
            workloads_polaris = json.load(f2)
        workload_util_ic2 = {}
        workload_util_polaris = {}
        all_workloads_ic2 = []
        all_workloads_polaris = []
        for wi in range(len(workloads_ic2)):
            workload_util_ic2[f"w{wi}"] = []
            tasks = workloads_ic2[wi]["tasklist"]
            for t in tasks:
                tid = t["task_id"]
                if tid is None:
                    continue
                gpu = t["gpus"]
                if gpu is None or gpu == 0:
                    continue
                for node in t["nodes"]:
                    metrics = node["metrics"]
                    gpu_utils = metrics["gpu_util"]
                    if not gpu_utils:
                        continue
                    for row in gpu_utils:
                        vals = row[1:]
                        for j in range(len(vals)):
                            workload_util_ic2[f"w{wi}"].append(float(vals[j]))
                            all_workloads_ic2.append(float(vals[j]))

        for wi in range(len(workloads_polaris)):
            workload_util_polaris[f"w{wi}"] = []
            
            tasks = workloads_polaris[wi]["tasklist"]
            for t in tasks:
                tid = t["task_id"]
                if tid is None:
                    continue
                gpu = t["gpus"]
                if gpu is None or gpu == 0:
                    continue
                for node in t["nodes"]:
                    metrics = node["metrics"]
                    gpu_utils = metrics["gpu_util"]
                    if not gpu_utils:
                        continue
                    for row in gpu_utils:
                        vals = row[1:]
                        for j in range(len(vals)):
                            workload_util_polaris[f"w{wi}"].append(float(vals[j]))
                            all_workloads_polaris.append(float(vals[j]))
        plot_cdf_gpu(all_workloads_ic2, all_workloads_polaris, p_tail=95, save_path=True, show=True)

'''
    plot_cdf_with_percentiles(
        cpu_nnls, cpu_gm, mem_nnls, mem_gm,
        p_tail=95,
        save_path=None,   # e.g., "results/decomp" to save PNGs
        show=True
    )
'''
