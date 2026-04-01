import json
import random
import copy
import numpy as np
from scipy.stats import poisson



def build_workloads(benchmark_data):
    all_jobs = benchmark_data

    NUM_WORKLOADS = 25
    JOBS_PER_WORKLOAD = 50
    MEAN_INTERARRIVAL = 10
    TIMESTEP = 5
    TOTAL_CPUS = 32

    all_workloads = []

    for wi in range(NUM_WORKLOADS):

        random.seed(wi)
        np.random.seed(wi)

        selected_jobs = random.sample(all_jobs, JOBS_PER_WORKLOAD)

        interarrivals = poisson.rvs(mu=MEAN_INTERARRIVAL, size=JOBS_PER_WORKLOAD)
        interarrivals = np.maximum(interarrivals, 1)

        arrival_times = np.cumsum(interarrivals)

        workload = {
            "workload_name": f"workload {wi}",
            "jobs": []
        }

        waiting_jobs = []
        running_jobs = []

        available_cpus = TOTAL_CPUS
        next_job_index = 0
        current_time = 0

        while next_job_index < JOBS_PER_WORKLOAD or waiting_jobs or running_jobs:

            next_arrival_time = (
                int(arrival_times[next_job_index])
                if next_job_index < JOBS_PER_WORKLOAD
                else float("inf")
            )

            next_finish_time = (
                min(job["end_time"] for job in running_jobs)
                if running_jobs
                else float("inf")
            )

            current_time = min(next_arrival_time, next_finish_time)
            if not running_jobs and waiting_jobs and next_job_index >= JOBS_PER_WORKLOAD:
                print("No running jobs, jobs still waiting, and no more arrivals.")
                print("First waiting job needs", waiting_jobs[0]["num_cpus"], "CPUs, available =", available_cpus)
                break
            # first release finished jobs
            finished_now = [job for job in running_jobs if job["end_time"] <= current_time]
            for job in finished_now:
                available_cpus += job["num_cpus"]
            running_jobs = [job for job in running_jobs if job["end_time"] > current_time]

            # then add newly arrived jobs to waiting queue
            while next_job_index < JOBS_PER_WORKLOAD and int(arrival_times[next_job_index]) <= current_time:
                job_copy = copy.deepcopy(selected_jobs[next_job_index])

                cpu_util = job_copy.get("cpu_util_percent", [])
                mem_util = job_copy.get("memory_util_percent", [])

                if len(cpu_util) > 0:
                    cpu_util = cpu_util[:-1]
                if len(mem_util) > 0:
                    mem_util = mem_util[:-1]

                num_cpus = int(job_copy.get("num_cpus", 0))
                if num_cpus <= 0:
                    num_cpus = 1
                if num_cpus > TOTAL_CPUS:
                    next_job_index += 1
                    continue

                waiting_jobs.append({
                    "job_id": next_job_index,
                    "benchmark": job_copy.get("benchmark", ""),
                    "num_cpus": num_cpus,
                    "threads_per_node": job_copy.get("threads_per_node", 0),
                    "arrival_time": int(arrival_times[next_job_index]),
                    "cpu_util_raw": cpu_util,
                    "memory_util_raw": mem_util
                })

                next_job_index += 1

            # FCFS: try to start waiting jobs in order
            started_any = True
            while started_any:
                started_any = False

                if not waiting_jobs:
                    break

                first_job = waiting_jobs[0]
                needed_cpus = first_job["num_cpus"]

                if needed_cpus <= available_cpus:
                    waiting_jobs.pop(0)

                    start_time = current_time
                    duration = len(first_job["cpu_util_raw"]) * TIMESTEP
                    end_time = start_time + duration

                    cpu_util_with_ts = []
                    mem_util_with_ts = []

                    for k in range(len(first_job["cpu_util_raw"])):
                        cpu_util_with_ts.append([
                            start_time + (k + 1) * TIMESTEP,
                            first_job["cpu_util_raw"][k]
                        ])

                    for k in range(len(first_job["memory_util_raw"])):
                        mem_util_with_ts.append([
                            start_time + (k + 1) * TIMESTEP,
                            first_job["memory_util_raw"][k]
                        ])

                    new_job = {
                        "job_id": first_job["job_id"],
                        "benchmark": first_job["benchmark"],
                        "num_cpus": first_job["num_cpus"],
                        "threads_per_node": first_job["threads_per_node"],
                        "arrival_time": first_job["arrival_time"],
                        "start_time": start_time,
                        "end_time": end_time,
                        "cpu_util": cpu_util_with_ts,
                        "memory_util": mem_util_with_ts
                    }

                    workload["jobs"].append(new_job)
                    running_jobs.append(new_job)
                    available_cpus -= needed_cpus
                    started_any = True

        all_workloads.append(workload)

    with open("benchmark_new.json", "w") as f:
        json.dump(all_workloads, f, indent=2)

    return all_workloads

if __name__ == "__main__":  
    with open("benchmark_results.json", "r") as f:
        benchmark_data = json.load(f)
    workloads = build_workloads(benchmark_data)
