import numpy as np
import matplotlib.pyplot as plt
import json
from algorithm_blockwise_nonnegative_least_squares import nnls_main
from algorithm_greedy_mean_only import gm_main

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


def plot_cdf_with_percentiles(
    cpu_mae_nnls,
    cpu_mae_greedy,
    mem_mae_nnls,
    mem_mae_greedy,
    type,
    *,
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

    cpu_nnls = _extract(cpu_mae_nnls)
    cpu_gm = _extract(cpu_mae_greedy)
    mem_nnls = _extract(mem_mae_nnls)
    mem_gm = _extract(mem_mae_greedy)

    # --- CPU CDF ---
    fig1 = plt.figure()
    x1, y1 = _cdf_xy(cpu_nnls)
    x2, y2 = _cdf_xy(cpu_gm)

    plt.plot(x1, y1, label="NNLS")
    plt.plot(x2, y2, label="Greedy Mean")

    # percentiles
    for label, values in [("NNLS", cpu_nnls), ("Greedy Mean", cpu_gm)]:
        p50 = _percentile(values, 50)
        pt = _percentile(values, p_tail)
        if np.isfinite(p50):
            plt.axvline(p50, linestyle="--", linewidth=1)
        if np.isfinite(pt):
            plt.axvline(pt, linestyle=":", linewidth=1)

    plt.title(f"{title_prefix}: CPU")
    plt.xlabel(xlabel_cpu)
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Print numbers you can cite in text
    cpu_p50_nnls, cpu_p95_nnls = _percentile(cpu_nnls, 50), _percentile(cpu_nnls, p_tail)
    cpu_p50_gm, cpu_p95_gm = _percentile(cpu_gm, 50), _percentile(cpu_gm, p_tail)
    print("CPU MAE percentiles")
    print(f"  NNLS:       p50={cpu_p50_nnls:.6g}, p{p_tail}={cpu_p95_nnls:.6g}")
    print(f"  GreedyMean: p50={cpu_p50_gm:.6g}, p{p_tail}={cpu_p95_gm:.6g}")
    if np.isfinite(cpu_p95_nnls) and cpu_p95_nnls > 0:
        print(f"  Tail ratio (GreedyMean/NNLS) at p{p_tail}: {cpu_p95_gm / cpu_p95_nnls:.3g}")

    # --- Memory CDF ---
    fig2 = plt.figure()
    x1, y1 = _cdf_xy(mem_nnls)
    x2, y2 = _cdf_xy(mem_gm)

    plt.plot(x1, y1, label="NNLS")
    plt.plot(x2, y2, label="Greedy Mean")

    for label, values in [("NNLS", mem_nnls), ("Greedy Mean", mem_gm)]:
        p50 = _percentile(values, 50)
        pt = _percentile(values, p_tail)
        if np.isfinite(p50):
            plt.axvline(p50, linestyle="--", linewidth=1)
        if np.isfinite(pt):
            plt.axvline(pt, linestyle=":", linewidth=1)

    plt.title(f"{title_prefix}: Memory")
    plt.xlabel(xlabel_mem)
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.legend()

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
        fig1.savefig(f"{save_path}_cpu_{type}_cdf.png", dpi=300, bbox_inches="tight")
        fig2.savefig(f"{save_path}_mem_{type}_cdf.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return {
        "cpu": {"nnls_p50": cpu_p50_nnls, f"nnls_p{p_tail}": cpu_p95_nnls,
                "gm_p50": cpu_p50_gm, f"gm_p{p_tail}": cpu_p95_gm},
        "mem": {"nnls_p50": mem_p50_nnls, f"nnls_p{p_tail}": mem_p95_nnls,
                "gm_p50": mem_p50_gm, f"gm_p{p_tail}": mem_p95_gm},
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
        cpu_nnls = nnls_main(workloads, system_loads, utilization="cpu")
        mem_nnls = nnls_main(workloads, system_loads, utilization="memory")
        cpu_gm = gm_main(workloads, system_loads, utilization="cpu")
        mem_gm = gm_main(workloads, system_loads, utilization="memory")
        plot_cdf_with_percentiles(cpu_nnls, cpu_gm, mem_nnls, mem_gm, type="ic2", p_tail=95, save_path=True, show=True)
       

    elif prompt == "polaris":
        workload_file = input("Enter Polaris workload file: ")
        system_file = input("Enter Polaris system load file: ")
        with open(workload_file, "r") as f1:
            workloads = json.load(f1)
        with open(system_file, "r") as f2:
            system_loads = json.load(f2)
        cpu_nnls = nnls_main(workloads, system_loads, utilization="cpu")
        mem_nnls = nnls_main(workloads, system_loads, utilization="memory")
        cpu_gm = gm_main(workloads, system_loads, utilization="cpu")
        mem_gm = gm_main(workloads, system_loads, utilization="memory")
        plot_cdf_with_percentiles(cpu_nnls, cpu_gm, mem_nnls, mem_gm, type="polaris", p_tail=95, save_path=True, show=True)


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
        cpu_nnls_ic2 = nnls_main(workloads_ic2, system_loads_ic2, utilization="cpu")
        mem_nnls_ic2 = nnls_main(workloads_ic2, system_loads_ic2, utilization="memory")
        cpu_gm_ic2 = gm_main(workloads_ic2, system_loads_ic2, utilization="cpu")
        mem_gm_ic2 = gm_main(workloads_ic2, system_loads_ic2, utilization="memory")
        cpu_nnls_polaris = nnls_main(workloads_polaris, system_loads_polaris, utilization="cpu")
        mem_nnls_polaris = nnls_main(workloads_polaris, system_loads_polaris, utilization="memory")
        cpu_gm_polaris = gm_main(workloads_polaris, system_loads_polaris, utilization="cpu")
        mem_gm_polaris = gm_main(workloads_polaris, system_loads_polaris, utilization="memory")
        plot_cdf_with_percentiles(cpu_nnls_ic2, cpu_gm_ic2, mem_nnls_ic2, mem_gm_ic2, type="ic2", p_tail=95, save_path=True, show=True)
        plot_cdf_with_percentiles(cpu_nnls_polaris, cpu_gm_polaris, mem_nnls_polaris, mem_gm_polaris, type="polaris", p_tail=95, save_path=True, show=True)

'''
    plot_cdf_with_percentiles(
        cpu_nnls, cpu_gm, mem_nnls, mem_gm,
        p_tail=95,
        save_path=None,   # e.g., "results/decomp" to save PNGs
        show=True
    )
'''
