from algorithm_blockwise_nonnegative_least_squares import pertask_utilization_NNLS
from algorithm_greedy_mean_only import pertask_utilization_greedy
import json
import pandas as pd

import json
import pandas as pd

def check(system_file, workloads_file, utilization, tid):
    contribs_nnls = pertask_utilization_NNLS(system_file, workloads_file, utilization)
    rows = []
    for (wi, node), task_list in contribs_nnls.items():
        if wi != "w0":
            break
        for task in task_list:
            if task["task_id"] == tid:
                utils = [u for _, u in task["Util"]]
                print(f"workload={wi}, node={node}, tid={tid} utils:")
                print(utils)

    
def exclude_short_tasks(workloads_file):
    """
    contribs: dict {(wi, node): [task_dict, ...]}
    Returns: set of (wi, node, tid) to exclude
    """
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
                print(f"Short task: workload={wi}, tid={tid}, dur={dur}")
                if f"w{wi}" not in short:
                    short[f"w{wi}"] = []
                short[f"w{wi}"].append(tid)
        #print(f"Workload {wi} has {len(short.get(wi, []))} short tasks.")
    #print(short.values())
    return short

def _imbalance(system_file, workloads_file, utilization, exclude=False):
    contribs_nnls = pertask_utilization_NNLS(system_file, workloads_file, utilization)
    contribs_greedy = pertask_utilization_greedy(system_file, workloads_file, utilization)
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



    #print(f"exclude: {exclude}")

    ti_wi_tid_nnls= {}
    ti_wi_tid_gm = {}
    si_wi_tid_nnls = {}
    si_wi_tid_gm = {}
    
    
    for (wi, node), task_list in contribs_nnls.items():
        #task_list = list of dict
        #print(wi)
        
        if wi not in ti_wi_tid_nnls:
            ti_wi_tid_nnls[wi] = {}
        if wi not in si_wi_tid_nnls:
            si_wi_tid_nnls[wi] = {}

        for task in task_list: #single task dict
            tid = task["task_id"]
            #print(f"tid={tid}, task={task}")
            #print(tid)
            if wi in exclude and tid in exclude[wi]:
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

    for wi in si_wi_tid_nnls:
        for tid in si_wi_tid_nnls[wi]:
            vals = si_wi_tid_nnls[wi][tid]
            max_util_across_node = max(vals)
            mean_util_of_all_max = sum(vals) / len(vals)
            si_wi_tid_nnls[wi][tid] = 1 - mean_util_of_all_max/max_util_across_node
            
    
    '''check again
    for wi, tid in ti_by_task_node_nnls.items():
        print(tid.items())
        print(len(tid))
        break
    '''
    
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

    return ti_wi_tid_nnls, ti_wi_tid_gm, si_wi_tid_nnls, si_wi_tid_gm
    

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

    ncheck = input("Check: ")
    if ncheck != "N":
        for i in range(51):
            check(system_ic2, workloads_ic2, "cpu", i)
    short_ic2 = exclude_short_tasks(workloads_ic2)

    '''
    ti_nnls_cpu_ic2, ti_gm_cpu_ic2, si_nnls_cpu_ic2, si_gm_cpu_ic2 = _imbalance(system_ic2, workloads_ic2, "cpu")
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