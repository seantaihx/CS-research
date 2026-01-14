import json
import numpy as np
from pathlib import Path
import pandas as pd



def make_table(system_file, workload_file):
    system_all = json.load(open(system_file))
    workload_all = json.load(open(workload_file))
    assert len(system_all) == len(workload_all)

    # One value per workload
    all_nodes = []
    all_durations = []
    all_cpu_peaks = []
    all_mem_peaks = []

    # Loop through workloads
    for wi in range(len(system_all)):
        system_entry = system_all[wi]
        node_list = system_entry.get("node_list", [])

        num_nodes = len(node_list)

        node_durations = []
        node_cpu_peaks = []
        node_mem_peaks = []


        #num = 0
        # Loop through nodes
        for node in node_list:
            #num += 1
            metrics = node.get("metrics", {})

            cpu_pairs = metrics.get("cpu_util", [])
            mem_pairs = metrics.get("memory_util", [])

            # CPU is required
            if not cpu_pairs:
                continue

            # ---- CPU timestamps and values ----
            cpu_times = []
            cpu_values = []

            for t, v in cpu_pairs:

                cpu_times.append(float(t))
                cpu_values.append(float(v))

            # Duration and peak CPU for this node
            duration = cpu_times[-1] - cpu_times[0]
            cpu_peak = max(cpu_values)
            #print(cpu_peak, end = " ")
   
            node_durations.append(duration)
            node_cpu_peaks.append(cpu_peak)

            # ---- Memory peak (optional) ----
            if mem_pairs:
                mem_values = []

                for t, v in mem_pairs:
                    mem_values.append(float(v))

                mem_max = max(mem_values)
                print(mem_max, end = " ")
                node_mem_peaks.append(mem_max)
                

        #print(num)
        print()
        # Convert node-level → workload-level
        avg_duration = sum(node_durations) / len(node_durations)
        total_cpu_peak = max(node_cpu_peaks)

        if len(node_mem_peaks) > 0:
            total_mem_peak = max(node_mem_peaks)
        else:
            total_mem_peak = float("nan")

        # Save one value per workload
        all_nodes.append(num_nodes)
        all_durations.append(avg_duration)
        all_cpu_peaks.append(total_cpu_peak)
        all_mem_peaks.append(total_mem_peak)

    # ---- Summary statistics across workloads ----
    def stats(data):
        if len(data) == 0:
            return [float("nan")] * 4

        return [
            float(np.median(data)),
            float(np.mean(data)),
            float(np.max(data)),
            float(np.std(data)),
        ]

    summary_rows = [
        ["No. of nodes"] + stats(all_nodes),
        ["Duration"] + stats(all_durations),
        ["CPU util (peak)"] + stats(all_cpu_peaks),
        ["Memory util (peak)"] + stats(all_mem_peaks),
    ]

    df_summary = pd.DataFrame(
        summary_rows,
        columns=["Metrics", "Median", "Mean", "Max", "Std Dev"]
    )

    # Export (CSV works everywhere, Excel can open it)
    df_summary.to_csv("table_workload_polaris.csv", index=False)
    print("Saved")

    return df_summary


if __name__ == "__main__":
    system_file = Path("all_system_loads_polaris.json")
    workload_file = Path("all_workloads_polaris.json")
    #system_file = input("Enter system file path: ")
    #workload_file = input("Enter workload file path: ")
    table = make_table(system_file, workload_file)
    print(table)