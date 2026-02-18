import numpy as np
import matplotlib.pyplot as plt
from imbalance import _imbalance
import json
from imbalance import exclude_short_tasks

def to_1d_list(ti_nnls, ti_gm, si_nnls, si_gm):
    """
    Convert the nested dicts of imbalance factors into flat lists.
    """
    ti_values_nnls = []
    ti_values_gm = []
    si_values_nnls = []
    si_values_gm = []

    for wi in ti_nnls:
        for tid in ti_nnls[wi]:
            ti_values_nnls.append(ti_nnls[wi][tid][1])  # ti is the second element in the tuple (node, ti)
    for wi in ti_gm:
        for tid in ti_gm[wi]:
            ti_values_gm.append(ti_gm[wi][tid][1])
    for wi in si_nnls:
        for tid in si_nnls[wi]:
            si_values_nnls.append(si_nnls[wi][tid])
    for wi in si_gm:
        for tid in si_gm[wi]:
            si_values_gm.append(si_gm[wi][tid])
    
    return ti_values_nnls, ti_values_gm, si_values_nnls, si_values_gm

def plot_unweighted_pdf_cdf(
    values,
    bins=30,
    x_range=(0.0, 1.0),
    y_range=(0.0, 30.0),
    cuts=(0.2, 0.6),
    title="CDF/PDF of Imbalance Factor",
    xlabel="Imbalance Factor",
    save_path=None,
    ax_pdf=None,
    linestyle='-'
):
    """
    values: list or numpy array of imbalance values (typically 0–1)
    """

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    create = False
    if ax_pdf is None:
        fig, ax_pdf = plt.subplots()
        create = True
    # ------------------ PDF (histogram) ------------------
    hist, bin_edges = np.histogram(values, bins=bins, range=x_range, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = bin_edges[1:] - bin_edges[:-1]

    # ------------------ CDF ------------------
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)


    # PDF bars
    ax_pdf.bar(centers, hist, width=widths, alpha=0.35, label="PDF")
    ax_pdf.set_xlabel(xlabel)
    ax_pdf.set_ylabel("PDF (%)")

    # CDF line
    ax_cdf = ax_pdf.twinx()
    ax_cdf.plot(sorted_vals, cdf * 100, linewidth=2, label="CDF")
    ax_cdf.set_ylabel("CDF (%)")
    #ax_cdf.set_ylim(y_range)

    # vertical category lines
    for c in cuts:
        ax_pdf.axvline(c, linestyle=linestyle, linewidth=1)
        ax_cdf.axvline(c, linestyle=linestyle, linewidth=1)

    # annotate percentages at cuts
    for c in cuts:
        idx = np.searchsorted(sorted_vals, c, side="right") - 1
        pct = 0 if idx < 0 else cdf[idx] * 100
        ax_cdf.annotate(f"({c:.2f}, {pct:.1f}%)",
                        xy=(c, pct),
                        xytext=(5, 5),
                        textcoords="offset points")
    ax_pdf.set_ylim(y_range)
    ax_pdf.set_xlim(x_range)
    ax_pdf.set_title(title)

    # combine legends
    lines1, labels1 = ax_pdf.get_legend_handles_labels()
    lines2, labels2 = ax_cdf.get_legend_handles_labels()
    ax_cdf.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax_pdf.grid(True, alpha=0.25)

    if create:
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

def plot_imbalance_2x2(nnls_ic2_cpu, nnls_polaris_cpu, nnls_ic2_memory, nnls_polaris_memory,
                      big_title, xlabel, save_path):

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    plot_unweighted_pdf_cdf(nnls_ic2_cpu, title="NNLS - IC2 CPU Utilization", xlabel=xlabel, ax_pdf=axes[0, 0], linestyle='--')
    plot_unweighted_pdf_cdf(nnls_polaris_cpu, title="NNLS - Polaris CPU Utilization", xlabel=xlabel, ax_pdf=axes[0, 1], linestyle='--')
    plot_unweighted_pdf_cdf(nnls_ic2_memory, title="NNLS - IC2 Memory Utilization", xlabel=xlabel, ax_pdf=axes[1, 0], linestyle='--')
    plot_unweighted_pdf_cdf(nnls_polaris_memory, title="NNLS - Polaris Memory Utilization", xlabel=xlabel, ax_pdf=axes[1, 1], linestyle='--')
    fig.suptitle(big_title, fontsize=16)
    #plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# ---------------- Example ----------------
if __name__ == "__main__":
    # Example imbalance values

    with open ("all_workloads_ic2.json", "r") as f1:
        workloads_ic2 = json.load(f1)
    with open ("all_system_loads_ic2.json", "r") as f2:
        system_loads_ic2 = json.load(f2)
    with open ("all_workloads_polaris.json", "r") as f3:
        workloads_polaris = json.load(f3)
    with open ("all_system_loads_polaris.json", "r") as f4:
        system_loads_polaris = json.load(f4)

    short_ic2 = exclude_short_tasks(workloads_ic2)
    short_polaris = exclude_short_tasks(workloads_polaris)
        
    ti_nnls_cpu_ic2, ti_gm_cpu_ic2, si_nnls_cpu_ic2, si_gm_cpu_ic2 = _imbalance(system_loads_ic2, workloads_ic2, "cpu", short_ic2)
    ti_nnls_cpu_polaris, ti_gm_cpu_polaris, si_nnls_cpu_polaris, si_gm_cpu_polaris = _imbalance(system_loads_polaris, workloads_polaris, "cpu", short_polaris)
    ti_nnls_memory_ic2, ti_gm_memory_ic2, si_nnls_memory_ic2, si_gm_memory_ic2 = _imbalance(system_loads_ic2, workloads_ic2, "memory", short_ic2)
    ti_nnls_memory_polaris, ti_gm_memory_polaris, si_nnls_memory_polaris, si_gm_memory_polaris = _imbalance(system_loads_polaris, workloads_polaris, "memory", short_polaris)

    ti_nnls_cpu_ic2_list, ti_gm_cpu_ic2_list, si_nnls_cpu_ic2_list, si_gm_cpu_ic2_list = to_1d_list(ti_nnls_cpu_ic2, ti_gm_cpu_ic2, si_nnls_cpu_ic2, si_gm_cpu_ic2)
    ti_nnls_cpu_polaris_list, ti_gm_cpu_polaris_list, si_nnls_cpu_polaris_list, si_gm_cpu_polaris_list = to_1d_list(ti_nnls_cpu_polaris, ti_gm_cpu_polaris, si_nnls_cpu_polaris, si_gm_cpu_polaris)
    ti_nnls_memory_ic2_list, ti_gm_memory_ic2_list, si_nnls_memory_ic2_list, si_gm_memory_ic2_list = to_1d_list(ti_nnls_memory_ic2, ti_gm_memory_ic2, si_nnls_memory_ic2, si_gm_memory_ic2)
    ti_nnls_memory_polaris_list, ti_gm_memory_polaris_list, si_nnls_memory_polaris_list, si_gm_memory_polaris_list = to_1d_list(ti_nnls_memory_polaris, ti_gm_memory_polaris, si_nnls_memory_polaris, si_gm_memory_polaris)

    plot_imbalance_2x2(ti_nnls_cpu_ic2_list, ti_nnls_cpu_polaris_list,
                      ti_nnls_memory_ic2_list, ti_nnls_memory_polaris_list,
                      big_title="Temporal Imbalance",
                      xlabel="Temporal Imbalance Factor",
                      save_path="imbalance_t_cpu_nnls.png")
    
    
    plot_imbalance_2x2(si_nnls_cpu_ic2_list, si_nnls_cpu_polaris_list,
                      si_nnls_memory_ic2_list, si_nnls_memory_polaris_list,
                      big_title="Spatial Imbalance",
                      xlabel="Spatial Imbalance Factor",
                      save_path="imbalance_s_cpu_nnls.png")

    
 