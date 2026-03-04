from algorithm_blockwise_nonnegative_least_squares import pertask_utilization_NNLS
from algorithm_greedy_mean_only import pertask_utilization_greedy
import json
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import numpy as np

def exclude_gpu(workloads_file):
    gpu_tasks = {}

    for wi in range(len(workloads_file)):
        tasks = workloads_file[wi]["tasklist"]
        for t in tasks:
            tid = t["task_id"]
            if tid is None:
                continue
            gpu = t["gpus"]
            if gpu is None:
                continue
            if f"w{wi}" not in gpu_tasks:
                gpu_tasks[f"w{wi}"] = []
            if gpu > 0:
                gpu_tasks[f"w{wi}"].append(tid)
        print(f"Workload {wi} has {len(gpu_tasks[f'w{wi}'])} GPU tasks.")
    #print(gpu_tasks.values())
    return gpu_tasks

def exclude_short_tasks(workloads_file):
    short = {}

    for wi in range(len(workloads_file)):
        tasks = workloads_file[wi]["tasklist"]
        for t in tasks:
            tid = t["task_id"]
            if tid is None:
                continue
            s = float(t["start_time"])
            f = float(t["finish_time"])
            if s is None or f is None:
                continue
            dur = f - s
            if dur < 60:
                if f"w{wi}" not in short:
                    short[f"w{wi}"] = []
                short[f"w{wi}"].append(tid)
        #print(f"Workload {wi} has {len(short.get(wi, []))} short tasks.")
    #print(short.values())
    return short

def _imbalance(system_file, workloads_file, utilization, exclude, include_cpu_hours, gpu_tasks):
    contribs_nnls_cpu = pertask_utilization_NNLS(system_file, workloads_file, utilization)
    contribs_greedy = pertask_utilization_greedy(system_file, workloads_file, utilization)
    contribs_nnls_memory = pertask_utilization_NNLS(system_file, workloads_file, "memory")
    '''
    contribs : dict
    {
    (wi, node) : [{
                "task_id": tid,
                "Node_name": node,
                "Util": [(timestamp, utilization), ...]
                },...]
    ...
    }
    '''

    ti_wi_tid_nnls= {}
    ti_wi_tid_gm = {}
    si_wi_tid_nnls = {}
    si_wi_tid_gm = {}
    cpu_max_wi_tid = {}
    mem_max_wi_tid = {}
    
    
    for (wi, node), task_list in contribs_nnls_cpu.items():
        #task_list = list of dict
        #print(wi)
        
        if wi not in ti_wi_tid_nnls:
            ti_wi_tid_nnls[wi] = {}
        if wi not in si_wi_tid_nnls:
            si_wi_tid_nnls[wi] = {}
        if wi not in cpu_max_wi_tid:
            cpu_max_wi_tid[wi] = {}
        if wi not in mem_max_wi_tid:
            mem_max_wi_tid[wi] = {}

        for task in task_list: #single task dict
            tid = task["task_id"]
            #print(f"tid={tid}, task={task}")
            #print(tid)

            if (wi in exclude and tid in exclude[wi]) or (wi in gpu_tasks and tid in gpu_tasks[wi]):
                #print(wi, tid)
                continue
                #print("Excluding short task:", wi)
                #print("Excluding short task:", wi, tid)
                
            utils = [u for _, u in task["Util"]]
            
            if not utils:
                continue

            max_util = max(utils)
            mean_util = sum(utils) / len(utils)
            ti = 1 - (mean_util/max_util)
            
            if tid not in ti_wi_tid_nnls[wi]:
                ti_wi_tid_nnls[wi][tid] = (node, ti)
            else:
                if ti > ti_wi_tid_nnls[wi][tid][1]:
                    #print(node, ti)
                    #print(ti_wi_tid_nnls[wi].values)
                    ti_wi_tid_nnls[wi][tid] = (node, ti)
                    

            if tid not in si_wi_tid_nnls[wi]:
                si_wi_tid_nnls[wi][tid] = []
                si_wi_tid_nnls[wi][tid].append(max_util)
            else:
                si_wi_tid_nnls[wi][tid].append(max_util)
                

            if tid not in cpu_max_wi_tid[wi]:
                cpu_max_wi_tid[wi][tid] = max_util
            else:
                if max_util > cpu_max_wi_tid[wi][tid]:
                    cpu_max_wi_tid[wi][tid] = max_util


    for wi in si_wi_tid_nnls:
        for tid in si_wi_tid_nnls[wi]:
            vals = si_wi_tid_nnls[wi][tid]
            max_util_across_node = max(vals)
            mean_util_of_all_max = sum(vals) / len(vals)
            si_wi_tid_nnls[wi][tid] = 1 - mean_util_of_all_max/max_util_across_node
   
    
    for (wi, node), task_list in contribs_nnls_memory.items():
        if wi not in mem_max_wi_tid:
            mem_max_wi_tid[wi] = {}
        for task in task_list:
            tid = task["task_id"]
            if (wi in exclude and tid in exclude[wi]) or (wi in gpu_tasks and tid in gpu_tasks[wi]):
                continue
            utils = [u for _, u in task["Util"]]
            if not utils:
                continue
            max_util = max(utils)
            if tid not in mem_max_wi_tid[wi]:
                mem_max_wi_tid[wi][tid] = max_util
            else:
                if max_util > mem_max_wi_tid[wi][tid]:
                    mem_max_wi_tid[wi][tid] = max_util
    


    for (wi, node), task_list in contribs_greedy.items():
        if wi not in ti_wi_tid_gm:
            ti_wi_tid_gm[wi] = {}
        if wi not in si_wi_tid_gm:
            si_wi_tid_gm[wi] = {}
        
        for task in task_list:
            tid = task["task_id"]
            utils = [u for _, u in task["Util"]]
            if not utils:
                continue
            max_util = max(utils)
            mean_util = sum(utils)/len(utils)
            ti = 1 - (mean_util/max_util)
            if tid not in ti_wi_tid_gm[wi]:
                ti_wi_tid_gm[wi][tid] = (node, ti)
            else:
                if ti > ti_wi_tid_gm[wi][tid][1]:
                    ti_wi_tid_gm[wi][tid] = (node, ti)

            if tid not in si_wi_tid_gm[wi]:
                si_wi_tid_gm[wi][tid] = []
                si_wi_tid_gm[wi][tid].append(max_util)
            else:
                si_wi_tid_gm[wi][tid].append(max_util)

    for wi in si_wi_tid_gm:
        for tid in si_wi_tid_gm[wi]:
            vals = si_wi_tid_gm[wi][tid]
            max_util_across_node = max(vals)
            mean_util_of_all_max = sum(vals)/len(vals)
            si_wi_tid_gm[wi][tid] = mean_util_of_all_max/max_util_across_node

    print("SI", si_wi_tid_nnls)

    return ti_wi_tid_nnls, ti_wi_tid_gm, si_wi_tid_nnls, si_wi_tid_gm, cpu_max_wi_tid, mem_max_wi_tid
    
 


def build_rows(ti_nnls, ti_gm, si_nnls, si_gm):
    rows = []
    for wi in ti_nnls:
        for tid in ti_nnls[wi]:
            row = {
                "workload": wi,
                "task_id": tid,
                "TI_NNLS": ti_nnls[wi][tid][1],   # (node, ti)
                "TI_GM":   ti_gm.get(wi, {}).get(tid, (None, None))[1],
                "SI_NNLS": si_nnls.get(wi, {}).get(tid, None),
                "SI_GM":   si_gm.get(wi, {}).get(tid, None),
            }
            rows.append(row)
    return rows



def cpu_hours(workloads_file):
    cpu_hours_wi_tid = {}
  
    for wi in range(len(workloads_file)):
        if f"w{wi}" not in cpu_hours_wi_tid:
            cpu_hours_wi_tid[f"w{wi}"] = {}
        tasks = workloads_file[wi]["tasklist"]
        for t in tasks:
            tid = t["task_id"]
            if tid is None:
                continue
            s = float(t["start_time"])
            f = float(t["finish_time"])
            if s is None or f is None:
                continue
            dur = f - s
            cpu = t["cpus"]
            cpu_hours = dur * cpu / 3600
            cpu_hours_wi_tid[f"w{wi}"][tid] = cpu_hours
    #print(cpu_hours_wi_tid)
    return cpu_hours_wi_tid


def overall_correlation(ti_nnls, si_nnls, cpu_max, mem_max, cpu_hour, cluster):
    ti_vals = []
    si_vals = []
    cpu_vals = []
    mem_vals = []
    cpu_hours_vals = []

    for wi in ti_nnls:
        for tid in ti_nnls[wi]:
            _, ti = ti_nnls[wi][tid]
            si = si_nnls[wi][tid]
            cpu = cpu_max[wi][tid]
            mem = mem_max[wi][tid]
            cpu_hours = cpu_hour[wi][tid]
            ti_vals.append(ti)
            si_vals.append(si)
            cpu_vals.append(cpu)
            mem_vals.append(mem)
            cpu_hours_vals.append(cpu_hours)

    pearson_ti_si = pearsonr(ti_vals, si_vals)[0]
    pearson_ti_cpu = pearsonr(ti_vals, cpu_vals)[0]
    pearson_ti_mem = pearsonr(ti_vals, mem_vals)[0]
    pearson_ti_cpu_hours = pearsonr(ti_vals, cpu_hours_vals)[0]
    pearson_si_cpu_hours = pearsonr(si_vals, cpu_hours_vals)[0]
    pearson_cpu_cpu_hours = pearsonr(cpu_vals, cpu_hours_vals)[0]
    pearson_mem_cpu_hours = pearsonr(mem_vals, cpu_hours_vals)[0]
    pearson_si_cpu = pearsonr(si_vals, cpu_vals)[0]
    pearson_si_mem = pearsonr(si_vals, mem_vals)[0]
    pearson_cpu_mem = pearsonr(cpu_vals, mem_vals)[0]

    spearman_ti_si = spearmanr(ti_vals, si_vals)[0]
    spearman_ti_cpu = spearmanr(ti_vals, cpu_vals)[0]
    spearman_ti_mem = spearmanr(ti_vals, mem_vals)[0]
    spearman_si_cpu = spearmanr(si_vals, cpu_vals)[0]
    spearman_si_mem = spearmanr(si_vals, mem_vals)[0]
    spearman_cpu_mem = spearmanr(cpu_vals, mem_vals)[0]
    spearman_cpu_cpu_hours = spearmanr(cpu_vals, cpu_hours_vals)[0]
    spearman_mem_cpu_hours = spearmanr(mem_vals, cpu_hours_vals)[0]
    spearman_ti_cpu_hours = spearmanr(ti_vals, cpu_hours_vals)[0]
    spearman_si_cpu_hours = spearmanr(si_vals, cpu_hours_vals)[0]

    kendalltau_ti_si = kendalltau(ti_vals, si_vals)[0]
    kendalltau_ti_cpu = kendalltau(ti_vals, cpu_vals)[0]
    kendalltau_ti_mem = kendalltau(ti_vals, mem_vals)[0]
    kendalltau_si_cpu = kendalltau(si_vals, cpu_vals)[0]
    kendalltau_si_mem = kendalltau(si_vals, mem_vals)[0]
    kendalltau_cpu_mem = kendalltau(cpu_vals, mem_vals)[0]
    kendalltau_ti_cpu_hours = kendalltau(ti_vals, cpu_hours_vals)[0]
    kendalltau_si_cpu_hours = kendalltau(si_vals, cpu_hours_vals)[0]
    kendalltau_cpu_cpu_hours = kendalltau(cpu_vals, cpu_hours_vals)[0]
    kendalltau_mem_cpu_hours = kendalltau(mem_vals, cpu_hours_vals)[0]

    def plot_heatmap(ti_si, ti_cpu, ti_mem, ti_cpu_hours, si_cpu, si_mem, si_cpu_hours, cpu_mem, cpu_cpu_hours, mem_cpu_hours, type):
        labels = ["TI", "SI", "CPU", "Memory", "CPU Hours"]
        corr_matrix = np.array([
            [1, ti_si, ti_cpu, ti_mem, ti_cpu_hours],
            [ti_si, 1, si_cpu, si_mem, si_cpu_hours],
            [ti_cpu, si_cpu, 1, cpu_mem, cpu_cpu_hours],
        [ti_mem, si_mem, cpu_mem, 1, mem_cpu_hours],
        [ti_cpu_hours, si_cpu_hours, cpu_cpu_hours, mem_cpu_hours, 1]
        ])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.xaxis.tick_top()
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
        plt.title(f"Overall Correlation Heatmap for {type}")
        plt.tight_layout()
        plt.savefig(f"{cluster}_{type}_overall_correlation_heatmap.png")
        plt.close()
    plot_heatmap(pearson_ti_si, pearson_ti_cpu, pearson_ti_mem, pearson_ti_cpu_hours,
                 pearson_si_cpu, pearson_si_mem, pearson_si_cpu_hours,
                 pearson_cpu_mem, pearson_cpu_cpu_hours, pearson_mem_cpu_hours, "Pearson")
    plot_heatmap(spearman_ti_si, spearman_ti_cpu, spearman_ti_mem, spearman_ti_cpu_hours,
                 spearman_si_cpu, spearman_si_mem, spearman_si_cpu_hours,
                 spearman_cpu_mem, spearman_cpu_cpu_hours, spearman_mem_cpu_hours, "Spearman")
    plot_heatmap(kendalltau_ti_si, kendalltau_ti_cpu, kendalltau_ti_mem, kendalltau_ti_cpu_hours,
                 kendalltau_si_cpu, kendalltau_si_mem, kendalltau_si_cpu_hours,
                 kendalltau_cpu_mem, kendalltau_cpu_cpu_hours, kendalltau_mem_cpu_hours, "Kendall Tau")
    
def correlation_per_workload(ti_nnls, si_nnls, cpu_max, mem_max, cpu_hour, cluster):
    ti_vals = {}
    si_vals = {}
    cpu_vals = {}
    mem_vals = {}
    cpu_hours_vals = {}

    pearson_ti_si = {}
    pearson_ti_cpu = {}
    pearson_ti_mem = {}
    pearson_ti_cpu_hours = {}
    pearson_si_cpu = {}
    pearson_si_mem = {}
    pearson_cpu_mem = {}
    pearson_si_cpu_hours = {}
    pearson_cpu_cpu_hours = {}
    pearson_mem_cpu_hours = {}
    
    spearman_ti_si = {}
    spearman_ti_cpu = {}
    spearman_ti_mem = {}
    spearman_si_cpu = {}
    spearman_si_mem = {}
    spearman_cpu_mem = {}
    spearman_ti_cpu_hours = {}
    spearman_si_cpu_hours = {}
    spearman_cpu_cpu_hours = {}
    spearman_mem_cpu_hours = {}

    kendalltau_ti_si = {}
    kendalltau_ti_cpu = {}
    kendalltau_ti_mem = {}
    kendalltau_si_cpu = {}
    kendalltau_si_mem = {}
    kendalltau_cpu_mem = {}
    kendalltau_ti_cpu_hours = {}
    kendalltau_si_cpu_hours = {}
    kendalltau_cpu_cpu_hours = {}
    kendalltau_mem_cpu_hours = {}
    for wi in ti_nnls:
        ti_vals[wi] = []
        si_vals[wi] = []
        cpu_vals[wi] = []
        mem_vals[wi] = []
        cpu_hours_vals[wi] = []

        for tid in ti_nnls[wi]:
            _, ti = ti_nnls[wi][tid]
            si = si_nnls[wi][tid]
            cpu = cpu_max[wi][tid]
            mem = mem_max[wi][tid]
            cpu_hours = cpu_hour[wi][tid]
            ti_vals[wi].append(ti)
            si_vals[wi].append(si)
            cpu_vals[wi].append(cpu)
            mem_vals[wi].append(mem)
            cpu_hours_vals[wi].append(cpu_hours)
        #print(len(ti_vals), len(si_vals), len(cpu_vals), len(mem_vals), len(cpu_hours_vals))
        
        
        pearson_ti_si[wi] = pearsonr(ti_vals[wi], si_vals[wi])[0]
        pearson_ti_cpu[wi] = pearsonr(ti_vals[wi], cpu_vals[wi])[0]
        pearson_ti_mem[wi] = pearsonr(ti_vals[wi], mem_vals[wi])[0]
        pearson_ti_cpu_hours[wi] = pearsonr(ti_vals[wi], cpu_hours_vals[wi])[0]
        pearson_si_cpu_hours[wi] = pearsonr(si_vals[wi], cpu_hours_vals[wi])[0]
        pearson_cpu_cpu_hours[wi] = pearsonr(cpu_vals[wi], cpu_hours_vals[wi])[0]
        pearson_mem_cpu_hours[wi] = pearsonr(mem_vals[wi], cpu_hours_vals[wi])[0]
        pearson_si_cpu[wi] = pearsonr(si_vals[wi], cpu_vals[wi])[0]
        pearson_si_mem[wi] = pearsonr(si_vals[wi], mem_vals[wi])[0]
        pearson_cpu_mem[wi] = pearsonr(cpu_vals[wi], mem_vals[wi])[0]
        
        spearman_ti_si[wi] = spearmanr(ti_vals[wi], si_vals[wi])[0]
        spearman_ti_cpu[wi] = spearmanr(ti_vals[wi], cpu_vals[wi])[0]
        spearman_ti_mem[wi] = spearmanr(ti_vals[wi], mem_vals[wi])[0]
        spearman_si_cpu[wi] = spearmanr(si_vals[wi], cpu_vals[wi])[0]
        spearman_si_mem[wi] = spearmanr(si_vals[wi], mem_vals[wi])[0]
        spearman_cpu_mem[wi] = spearmanr(cpu_vals[wi], mem_vals[wi])[0]
        spearman_cpu_cpu_hours[wi] = spearmanr(cpu_vals[wi], cpu_hours_vals[wi])[0]
        spearman_mem_cpu_hours[wi] = spearmanr(mem_vals[wi], cpu_hours_vals[wi])[0]
        spearman_ti_cpu_hours[wi] = spearmanr(ti_vals[wi], cpu_hours_vals[wi])[0]
        spearman_si_cpu_hours[wi] = spearmanr(si_vals[wi], cpu_hours_vals[wi])[0]

        kendalltau_ti_si[wi] = kendalltau(ti_vals[wi], si_vals[wi])[0]
        kendalltau_ti_cpu[wi] = kendalltau(ti_vals[wi], cpu_vals[wi])[0]
        kendalltau_ti_mem[wi] = kendalltau(ti_vals[wi], mem_vals[wi])[0]
        kendalltau_si_cpu[wi] = kendalltau(si_vals[wi], cpu_vals[wi])[0]
        kendalltau_si_mem[wi] = kendalltau(si_vals[wi], mem_vals[wi])[0]
        kendalltau_cpu_mem[wi] = kendalltau(cpu_vals[wi], mem_vals[wi])[0]
        kendalltau_ti_cpu_hours[wi] = kendalltau(ti_vals[wi], cpu_hours_vals[wi])[0]
        kendalltau_si_cpu_hours[wi] = kendalltau(si_vals[wi], cpu_hours_vals[wi])[0]
        kendalltau_cpu_cpu_hours[wi] = kendalltau(cpu_vals[wi], cpu_hours_vals[wi])[0]
        kendalltau_mem_cpu_hours[wi] = kendalltau(mem_vals[wi], cpu_hours_vals[wi])[0]
        
    def plot_heatmap(wi, type):
        labels = ["TI", "SI", "CPU", "Memory", "CPU Hours"]
        corr_matrix = np.array([
            [1,
             pearson_ti_si[wi], 
             pearson_ti_cpu[wi], 
             pearson_ti_mem[wi], 
             pearson_ti_cpu_hours[wi]],

            [pearson_ti_si[wi],
             1, 
             pearson_si_cpu[wi], 
             pearson_si_mem[wi], 
             pearson_si_cpu_hours[wi]],

            [pearson_ti_cpu[wi], 
             pearson_si_cpu[wi], 
             1, 
             pearson_cpu_mem[wi], 
             pearson_cpu_cpu_hours[wi]],

            [pearson_ti_mem[wi], 
             pearson_si_mem[wi], 
             pearson_cpu_mem[wi], 
             1, 
             pearson_mem_cpu_hours[wi]],

            [pearson_ti_cpu_hours[wi], 
             pearson_si_cpu_hours[wi], 
             pearson_cpu_cpu_hours[wi], 
             pearson_mem_cpu_hours[wi], 
             1]
        ])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.xaxis.tick_top()
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
        plt.title(f"{type} Correlation Heatmap for Workload {wi}")
        plt.tight_layout()
        plt.savefig(f"{cluster}_{type}_correlation_heatmap_{wi}.png")
        plt.close()
    
    for wi in ti_nnls:
        plot_heatmap(wi, "Pearson")
        plot_heatmap(wi, "Spearman")
        plot_heatmap(wi, "KendallTau")

if __name__ == "__main__":


    workloads_ic2_file = input("Enter workloads ic2 file: ")
    system_ic2_file = input("Enter system ic2 file: ")
    workloads_polaris_file = input("Enter workload polaris file: ")
    system_polaris_file = input("Enter system polaris file: ")
    with open (workloads_ic2_file, "r") as f1:
        workloads_ic2 = json.load(f1)
    with open (system_ic2_file, "r") as f2:
        system_ic2 = json.load(f2)
    with open (workloads_polaris_file, "r") as f3:
        workloads_polaris = json.load(f3)
    with open (system_polaris_file, "r") as f4:
        system_polaris = json.load(f4)

    short_ic2 = exclude_short_tasks(workloads_ic2)
    short_polaris = exclude_short_tasks(workloads_polaris)
    gpu_task_ic2 = exclude_gpu(workloads_ic2)
    gpu_task_polaris = exclude_gpu(workloads_polaris)

    cpu_hours_ic2 = cpu_hours(workloads_ic2)
    cpu_hours_polaris = cpu_hours(workloads_polaris)

    ti_nnls_cpu_ic2, ti_gm_cpu_ic2, si_nnls_cpu_ic2, si_gm_cpu_ic2, cpu_max_ic2, mem_max_ic2 = _imbalance(system_ic2, workloads_ic2, "cpu", exclude=short_ic2, include_cpu_hours=cpu_hours_ic2, gpu_tasks=gpu_task_ic2)
    ti_nnls_cpu_polaris, ti_gm_cpu_polaris, si_nnls_cpu_polaris, si_gm_cpu_polaris, cpu_max_polaris, mem_max_polaris = _imbalance(system_polaris, workloads_polaris, "cpu", exclude=short_polaris, include_cpu_hours=cpu_hours_polaris, gpu_tasks=gpu_task_polaris)


    correlation_per_workload(ti_nnls_cpu_ic2, si_nnls_cpu_ic2, cpu_max_ic2, mem_max_ic2, cpu_hours_ic2, "ic2")
    overall_correlation(ti_nnls_cpu_ic2, si_nnls_cpu_ic2, cpu_max_ic2, mem_max_ic2, cpu_hours_ic2, "ic2")
    correlation_per_workload(ti_nnls_cpu_polaris, si_nnls_cpu_polaris, cpu_max_polaris, mem_max_polaris, cpu_hours_polaris, "polaris")
    overall_correlation(ti_nnls_cpu_polaris, si_nnls_cpu_polaris, cpu_max_polaris, mem_max_polaris, cpu_hours_polaris, "polaris")
    '''
    ti_nnls_cpu_ic2, ti_gm_cpu_ic2, si_nnls_cpu_ic2, si_gm_cpu_ic2, cpu_max_ic2, mem_max_ic2 = _imbalance(system_ic2, workloads_ic2, "cpu")
    ti_nnls_cpu_polaris, ti_gm_cpu_polaris, si_nnls_cpu_polaris, si_gm_cpu_polaris = _imbalance(system_polaris, workloads_polaris, "cpu")
    ti_nnls_memory_ic2, ti_gm_memory_ic2, si_nnls_memory_ic2, si_gm_memory_ic2 = _imbalance(system_ic2, workloads_ic2, "memory")
    ti_nnls_memory_polaris, ti_gm_memory_polaris, si_nnls_memory_polaris, si_gm_memory_polaris = _imbalance(system_polaris, workloads_polaris, "memory")
    
    rows_cpu_ic2 = build_rows(ti_nnls_cpu_ic2, ti_gm_cpu_ic2, si_nnls_cpu_ic2, si_gm_cpu_ic2)
    rows_cpu_polaris = build_rows(ti_nnls_cpu_polaris, ti_gm_cpu_polaris, si_nnls_cpu_polaris, si_gm_cpu_polaris)
    rows_memory_ic2 = build_rows(ti_nnls_memory_ic2, ti_gm_memory_ic2, si_nnls_memory_ic2, si_gm_memory_ic2)
    rows_memory_polaris = build_rows(ti_nnls_memory_polaris, ti_gm_memory_polaris, si_nnls_memory_polaris, si_gm_memory_polaris)


    df_cpu_ic2 = pd.DataFrame(rows_cpu_ic2)
    df_cpu_polaris = pd.DataFrame(rows_cpu_polaris)
    df_memory_ic2 = pd.DataFrame(rows_memory_ic2)
    df_memory_polaris = pd.DataFrame(rows_memory_polaris)

    df_cpu_ic2.to_csv("Imbalance_cpu_ic2.csv", index = False)
    df_cpu_polaris.to_csv("Imbalance_cpu_polaris.csv", index = False)
    df_memory_ic2.to_csv("Imbalance_memory_ic2.csv", index=False)
    df_memory_polaris.to_csv("Imbalance_memory_polaris.csv", index=False)
    
    #print(df_cpu_ic2)
    #print()
    #print(df_cpu_polaris)
    #print()
    #print(df_memory_ic2)
    #print()
    #print(df_memory_polaris)
    '''