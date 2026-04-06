import json
import numpy as np
import pandas as pd


def stats(data):
    """Return [median, mean, max, std] for a list (nan-safe)."""
    if len(data) == 0:
        return [float("nan"), float("nan"), float("nan"), float("nan")]

    arr = np.array(data, dtype=float)
    return [
        float(np.nanmedian(arr)),
        float(np.nanmean(arr)),
        float(np.nanmax(arr)),
        float(np.nanstd(arr)),
    ]


def job_peak_gpu_util(task_nodes):
    """
    task_nodes: task["nodes"]
    Each node has metrics["gpu_util"] like:
      [ [timestamp, gpu0, gpu1, ...], ... ]
    For each timestamp row, average gpu0..gpuK, then take peak over time
    across the whole job (all nodes).
    """
    per_timestamp_avgs = []

    for tnode in task_nodes:
        metrics = tnode.get("metrics", {})
        gpu_util_list = metrics.get("gpu_util", [])

        for row in gpu_util_list:
            # row = [timestamp, gpu0, gpu1, ...]
            if row is None or len(row) < 2:
                continue

            gpu_vals = []
            for j in range(1, len(row)):
                try:
                    gpu_vals.append(float(row[j]))
                except:
                    pass

            if len(gpu_vals) == 0:
                continue

            per_timestamp_avgs.append(sum(gpu_vals) / len(gpu_vals))

    if len(per_timestamp_avgs) == 0:
        return float("nan")

    return max(per_timestamp_avgs)


def job_duration_seconds(task, task_nodes):
    """
    Prefer job-level duration from task["start_time"] and task["finish_time"].
    Fallback: use gpu_util timestamps (first/last row encountered).
    """
    # Preferred: start_time/finish_time
    try:
        start_t = float(task.get("start_time", "nan"))
        finish_t = float(task.get("finish_time", "nan"))
        if not (np.isnan(start_t) or np.isnan(finish_t)):
            return finish_t - start_t
    except:
        pass

    # Fallback: gpu_util timestamps
    timestamps = []
    for tnode in task_nodes:
        metrics = tnode.get("metrics", {})
        gpu_util_list = metrics.get("gpu_util", [])
        for row in gpu_util_list:
            if row is None or len(row) < 1:
                continue
            try:
                timestamps.append(float(row[0]))
            except:
                pass

    if len(timestamps) < 2:
        return float("nan")

    return max(timestamps) - min(timestamps)


def make_gpu_job_summary(workload_file, clusters):
    workload_all = json.load(open(workload_file))

    # Job-level lists (one value per GPU job)
    gpus_active_per_job = []
    gpu_duration_per_job = []
    gpu_peak_util_per_job = []

    gpu_job_count = 0

    for workload_entry in workload_all:
        task_list = workload_entry.get("tasklist", [])

        for task in task_list:
            gpus = task.get("gpus", 0)

            # Only GPU jobs
            if gpus == 0 or gpus == 0.0:
                continue

            gpu_job_count += 1
            task_nodes = task.get("nodes", [])

            # 1) GPUs active (job-level)
            gpus_active_per_job.append(float(gpus))

            # 2) Duration (job-level)
            gpu_duration_per_job.append(job_duration_seconds(task, task_nodes))

            # 3) Peak GPU util (job-level)
            gpu_peak_util_per_job.append(job_peak_gpu_util(task_nodes))

    # Build your table: rows are metrics, cols are stats
    summary_rows = [
        ["GPUs active (per job)"] + stats(gpus_active_per_job),
        ["GPU duration (per job, sec)"] + stats(gpu_duration_per_job),
        ["GPU util peak (per job)"] + stats(gpu_peak_util_per_job),
    ]
    #print("Total GPU jobs:", gpu_job_count)
    df = pd.DataFrame(summary_rows, columns=["Metrics", "Median", "Mean", "Max", "Std Dev"])
    df.to_csv(f"table_GPU_{clusters}.csv", index=False)
    print("Saved:", f"table_GPU_{clusters}.csv")
    return df


if __name__ == "__main__":
    workload_polaris_file = "all_workloads_polaris.json"  # change path if needed
    workload_ic2_file = "all_workloads_ic2.json"  # change path if needed
    df_polaris = make_gpu_job_summary(workload_polaris_file, "polaris")
    df_ic2 = make_gpu_job_summary(workload_ic2_file, "ic2")
    print(df_polaris)
    print(df_ic2)