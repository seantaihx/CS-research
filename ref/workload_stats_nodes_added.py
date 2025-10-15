import matplotlib.pyplot as plt
import json
import numpy as np
POLARIS_PATH = "../data/polaris/"
IC2_PATH = "../data/ic2/"
AWS_PATH = "../data/aws/"
def visualize_runtime_cpus_gpus(path, cluster_name, axs,index):
    small_runtimes = []
    small_cpus = []
    small_gpus = []
    scattered_runtimes = []
    scattered_cpus = []
    scattered_gpus = []
    mixed_runtimes = []
    mixed_cpus = []
    mixed_gpus = []
    scalable_runtimes = []
    scalable_cpus = []
    scalable_gpus = []
    threaded_runtimes = []
    threaded_cpus = []
    threaded_gpus = []
    with open(path, 'r') as f:
        workloads= json.load(f)
    count = 0
    for workload in workloads:
        if "small" in workload["workload-name"]:
            tasklist = workload["tasklist"]
            for task in tasklist:
                nodelist = task["nodes"]
                for node in nodelist:
                    nodename = node["node_name"]
                rt = (float(task["finish_time"]) - float(task["start_time"]))
                small_runtimes.append(rt)
                small_cpus.append(float(task["cpus"]))
                small_gpus.append(float(task["gpus"]))
        elif "scattered" in workload["workload-name"]:
            tasklist = workload["tasklist"]
            for task in tasklist:
                rt = (float(task["finish_time"]) - float(task["start_time"]))
                scattered_runtimes.append(rt)
                scattered_cpus.append(float(task["cpus"]))
                scattered_gpus.append(float(task["gpus"]))
        elif "mixed" in workload["workload-name"]:
            tasklist = workload["tasklist"]
            for task in tasklist:
                rt = (float(task["finish_time"]) - float(task["start_time"]))
                mixed_runtimes.append(rt)
                mixed_cpus.append(float(task["cpus"]))
                mixed_gpus.append(float(task["gpus"]))
        elif "scalable" in workload["workload-name"]:
            tasklist = workload["tasklist"]
            for task in tasklist:
                rt = (float(task["finish_time"]) - float(task["start_time"]))
                scalable_runtimes.append(rt)
                scalable_cpus.append(float(task["cpus"]))
                scalable_gpus.append(float(task["gpus"]))
        elif "threaded" in workload["workload-name"]:
            tasklist = workload["tasklist"]
            for task in tasklist:
                rt = (float(task["finish_time"]) - float(task["start_time"]))
                threaded_runtimes.append(rt)
                threaded_cpus.append(float(task["cpus"]))
                threaded_gpus.append(float(task["gpus"]))
    system_runtimes = small_runtimes + scattered_runtimes + mixed_runtimes + scalable_runtimes + threaded_runtimes
    print("Mean:", np.array(system_runtimes).mean())
    y_axis_runtime = [small_runtimes, scattered_runtimes, mixed_runtimes, scalable_runtimes, threaded_runtimes]
    axs[index,0].set_yscale('log')
    axs[index,0].boxplot(y_axis_runtime, labels = x_axis)
    axs[index,0].set_xlabel("Workload Type")
    axs[index,0].set_ylabel("Minutes")
    axs[index,0].set_xticks(X_axis_len)
    axs[index,0].set_xticklabels(x_axis, fontsize=10, rotation=30)
    axs[index,0].set_title(cluster_name+" Cluster - Runtime vs. Workload Type",fontsize=10)
    
    y_axis_cpus = [small_cpus, scattered_cpus, mixed_cpus, scalable_cpus, threaded_cpus]
    #axs[0].set_yscale('log')
    axs[index,1].boxplot(y_axis_cpus, labels = x_axis)
    axs[index,1].set_xlabel("Workload Type")
    axs[index,1].set_ylabel("#CPUs")
    axs[index,1].set_xticks(X_axis_len)
    axs[index,1].set_xticklabels(x_axis, fontsize=10, rotation=30)
    axs[index,1].set_title(cluster_name+" Cluster - #CPUs vs. Workload Type",fontsize=10)
    
    y_axis_gpus = [small_gpus, scattered_gpus, mixed_gpus, scalable_gpus, threaded_gpus]
    #axs[0].set_yscale('log')
    axs[index,2].boxplot(y_axis_gpus, labels = x_axis)
    axs[index,2].set_xlabel("Workload Type")
    axs[index,2].set_ylabel("#GPUs")
    axs[index,2].set_xticks(X_axis_len)
    axs[index,2].set_xticklabels(x_axis, fontsize=10, rotation=30)
    axs[index,2].set_title(cluster_name+" Cluster - #GPUs vs. Workload Type",fontsize=10)
    return axs
    #plt.show()

if __name__ == "__main__":

    x_axis = ["small", "scattered", "mixed", "scalable", "threaded"]
    X_axis_len = np.arange(len(x_axis))
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    #fig.suptitle("", fontsize=12)
    path_list = ["../data/all_workloads_ic2.json","../data/all_workloads_polaris.json","../data/all_workloads_aws.json"]
    cluster_name = ["IC2", "Polaris", "AWS"]
    for i in range(3):
        axs = visualize_runtime_cpus_gpus(path_list[i], cluster_name[i], axs, i)
    fig.tight_layout()
    plt.show()
    #plt.savefig("workload_stats.png", dpi=300)
