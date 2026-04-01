import json
from cpu_imbalance_correlation import exclude_short_tasks
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import numpy as np

def gpu_imbalance(workload_file, short):
    gpu_imbalance = {}
    
    for wi in range(len(workload_file)):
        gpu_imbalance[f"w{wi}"] = {}
        tasks = workload_file[wi]["tasklist"]
        for t in tasks:
            tid = t["task_id"]
            if tid is None:
                continue
            if tid in short[f"w{wi}"]:
                continue
            gpu = t["gpus"]
            if gpu is None or gpu == 0:
                continue

            per_gpu_ti = []
            per_gpu_max = []

            for node in t["nodes"]:
                metrics = node["metrics"]
                gpu_utils = metrics["gpu_util"]
                if not gpu_utils:
                    continue
                gpu_series = []
                for i in range(int(gpu)):
                    gpu_series.append([])
                for row in gpu_utils:
                    vals = row[1:]
                    for j in range(len(vals)):
                        gpu_series[j].append(float(vals[j]))
                for series in gpu_series:
                    if not series:
                        continue
                    avg = sum(series) / len(series)
                    mx = max(series)

                    if mx == 0:
                        ti = 0
                    else:
                        ti = 1 - (avg / mx)
                    per_gpu_ti.append(ti)
                    per_gpu_max.append(mx)
            if not per_gpu_max:
                continue
            gpu_ti = max(per_gpu_ti)
            mean_max = sum(per_gpu_max) / len(per_gpu_max)
            max_max = max(per_gpu_max)
            gpu_si = 1 - (mean_max / max_max)

            gpu_imbalance[f"w{wi}"][tid] = (gpu_ti, gpu_si)
    return gpu_imbalance

def max_util_and_duration(workload_file, short):
    max_cpu = []
    max_mem = []
    duration = []
    for wi in range(len(workload_file)):
        tasks = workload_file[wi]["tasklist"]
        for t in tasks:
            tid = t["task_id"]
            if tid is None:
                continue
            if tid in short[f"w{wi}"]:
                continue
            if not t["gpus"] or t["gpus"] == 0:
                continue
            start = float(t["start_time"])
            finish = float(t["finish_time"])
            dur = finish - start
            duration.append(dur)
            cpu_utils = []
            mem_utils = []
            for node in t["nodes"]:
                metrics = node["metrics"]
                cpu_util = metrics["cpu_util"]
                mem_util = metrics["gpu_util"]
                if not cpu_util or not mem_util:
                    continue
                for row in cpu_util:
                    cpu_utils.append(float(row[1]))
                for row in mem_util:
                    mem_utils.append(max([float(x) for x in row[1:]]))
                    
            if not cpu_utils or not mem_utils:
                continue
            max_cpu_util = max(cpu_utils)
            max_mem_util = max(mem_utils)
            max_cpu.append(max_cpu_util)
            max_mem.append(max_mem_util)
    return max_cpu, max_mem, duration
            
def correlation_heatmap(imbalances, max_cpu, max_mem, duration, cluster):
    one_d_ti = []
    one_d_si = []
    for wi in range(len(imbalances)):
        for tid in imbalances[f"w{wi}"]:
            one_d_ti.append(imbalances[f"w{wi}"][tid][0])
            one_d_si.append(imbalances[f"w{wi}"][tid][1])
    print(len(one_d_ti), " ")
    print(len(duration), " ")
    print(len(max_cpu), " ")
    print(len(max_mem), " ")
    pearson_ti_si = pearsonr(one_d_ti, one_d_si)[0]
    pearson_ti_cpu = pearsonr(one_d_ti, max_cpu)[0]
    pearson_ti_mem = pearsonr(one_d_ti, max_mem)[0]
    pearson_ti_dur = pearsonr(one_d_ti, duration)[0]
    pearson_si_cpu = pearsonr(one_d_si, max_cpu)[0]
    pearson_si_mem = pearsonr(one_d_si, max_mem)[0]
    pearson_si_dur = pearsonr(one_d_si, duration)[0]
    pearson_cpu_dur = pearsonr(max_cpu, duration)[0]
    pearson_mem_dur = pearsonr(max_mem, duration)[0]
    pearson_cpu_mem = pearsonr(max_cpu, max_mem)[0]

    spearman_ti_si = spearmanr(one_d_ti, one_d_si)[0]
    spearman_ti_cpu = spearmanr(one_d_ti, max_cpu)[0]
    spearman_ti_mem = spearmanr(one_d_ti, max_mem)[0]
    spearman_ti_dur = spearmanr(one_d_ti, duration)[0]
    spearman_si_cpu = spearmanr(one_d_si, max_cpu)[0]
    spearman_si_mem = spearmanr(one_d_si, max_mem)[0]
    spearman_si_dur = spearmanr(one_d_si, duration)[0]
    spearman_cpu_dur = spearmanr(max_cpu, duration)[0]
    spearman_mem_dur = spearmanr(max_mem, duration)[0]
    spearman_cpu_mem = spearmanr(max_cpu, max_mem)[0]

    kendall_ti_si = kendalltau(one_d_ti, one_d_si)[0]
    kendall_ti_cpu = kendalltau(one_d_ti, max_cpu)[0]
    kendall_ti_mem = kendalltau(one_d_ti, max_mem)[0]
    kendall_ti_dur = kendalltau(one_d_ti, duration)[0]
    kendall_si_cpu = kendalltau(one_d_si, max_cpu)[0]
    kendall_si_mem = kendalltau(one_d_si, max_mem)[0]
    kendall_si_dur = kendalltau(one_d_si, duration)[0]
    kendall_cpu_dur = kendalltau(max_cpu, duration)[0]
    kendall_mem_dur = kendalltau(max_mem, duration)[0]
    kendall_cpu_mem = kendalltau(max_cpu, max_mem)[0]
    
    def plot_heatmap(ti_si, ti_cpu, ti_mem, ti_dur, si_cpu, si_mem, si_dur, cpu_dur, mem_dur, cpu_mem, type, cluster):
        labels = ["TI", "SI", "CPU", "Memory", "Duration"]
        
        corr_matrix = np.array([
            [1.0, ti_si, ti_cpu, ti_mem, ti_dur],
            [ti_si, 1.0, si_cpu, si_mem, si_dur],
            [ti_cpu, si_cpu, 1.0, cpu_mem, cpu_dur],
            [ti_mem, si_mem, cpu_mem, 1.0, mem_dur],
            [ti_dur, si_dur, cpu_dur, mem_dur, 1.0]
        ])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.xaxis.tick_top()

        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
        plt.title(f"Overall GPU Correlation Heatmap for {type} - Cluster {cluster}")
        plt.tight_layout()
        plt.savefig(f"gpu_correlation_heatmap_{type}_{cluster}.png")
        plt.close()

    plot_heatmap(pearson_ti_si, pearson_ti_cpu, pearson_ti_mem, pearson_ti_dur, pearson_si_cpu, pearson_si_mem, pearson_si_dur, pearson_cpu_dur, pearson_mem_dur, pearson_cpu_mem, "Pearson", cluster)
    plot_heatmap(spearman_ti_si, spearman_ti_cpu, spearman_ti_mem, spearman_ti_dur, spearman_si_cpu, spearman_si_mem, spearman_si_dur, spearman_cpu_dur, spearman_mem_dur, spearman_cpu_mem, "Spearman", cluster)
    plot_heatmap(kendall_ti_si, kendall_ti_cpu, kendall_ti_mem, kendall_ti_dur, kendall_si_cpu, kendall_si_mem, kendall_si_dur, kendall_cpu_dur, kendall_mem_dur, kendall_cpu_mem, "Kendall", cluster)

if __name__ == "__main__":
    with open("all_workloads_ic2.json", "r") as f:
        workload_file_ic2 = json.load(f)
    with open("all_workloads_polaris.json", "r") as f:
        workload_file_polaris = json.load(f)
    excluded_tasks_ic2 = exclude_short_tasks(workload_file_ic2)
    excluded_tasks_polaris = exclude_short_tasks(workload_file_polaris)
    max_cpu_ic2, max_mem_ic2, duration_ic2 = max_util_and_duration(workload_file_ic2, excluded_tasks_ic2)
    max_cpu_polaris, max_mem_polaris, duration_polaris = max_util_and_duration(workload_file_polaris, excluded_tasks_polaris)
    imbalance_ic2 = gpu_imbalance(workload_file_ic2, excluded_tasks_ic2)
    imbalance_polaris = gpu_imbalance(workload_file_polaris, excluded_tasks_polaris)

    correlation_heatmap(imbalance_ic2, max_cpu_ic2, max_mem_ic2, duration_ic2, "IC2")
    correlation_heatmap(imbalance_polaris, max_cpu_polaris, max_mem_polaris, duration_polaris, "Polaris")