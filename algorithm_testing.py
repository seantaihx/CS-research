import json
import numpy as np
import matplotlib.pyplot as plt
from algorithm_blockwise_nonnegative_least_squares import mean_then_blockwise_nnls


def NNLS_benchmark(benchmark,block_size=30, l2=1e-2):
    contribs_all = {} # estimated contributions
    benchmark_result = {} # metrics

    for wi, workload_entry in enumerate(benchmark):
        wname = workload_entry.get("workload_name", f"w{wi}")
        contribs_all[wname] = {}
        benchmark_result[wname] = {}

        for utilization in ["cpu", "memory"]:
            actual_task_contribs = {} # actual job_id: {"timestamps": ts, "util": vals}
            tasks = [] #actual job_id, start, finish
            all_ts = [] # all timestamps

            for job in workload_entry.get("jobs", []):
                job_id = int(job["job_id"])

                if utilization == "cpu":
                    pairs = job.get("cpu_util", [])
                else:
                    pairs = job.get("memory_util", [])

                if not pairs:
                    continue

                ts, vals = zip(*pairs)
                ts = np.array(list(map(float, ts)))
                vals = np.array(list(map(float, vals)))

                order = np.argsort(ts)
                ts = ts[order]
                vals = vals[order]

                actual_task_contribs[job_id] = {
                    "timestamps": ts,
                    "util": vals
                }

                start = float(job.get("start_time", ts[0]))
                finish = float(job.get("end_time", ts[-1]))

                tasks.append({
                    "task_id": job_id,
                    "start": start,
                    "finish": finish,
                })

                all_ts.extend(ts.tolist())

            if not tasks:
                continue

            union_ts = np.array(sorted(set(all_ts)), dtype=float)
            observed = np.zeros(len(union_ts)) #total observed over workload

            aligned_actual = {} #each task actual util aligned to union_ts
            for tid, data in actual_task_contribs.items():
                ts = data["timestamps"]
                vals = data["util"]

                arr = np.zeros(len(union_ts))
                ts_to_val = {float(t): float(v) for t, v in zip(ts, vals)}

                for i, t in enumerate(union_ts):
                    if t in ts_to_val:
                        arr[i] = ts_to_val[t]

                aligned_actual[tid] = arr
                observed += arr
            
            contribs = mean_then_blockwise_nnls(union_ts, observed, tasks, block_size, l2)

            contribs_all[wname][utilization] = contribs
            benchmark_result[wname][utilization] = {}
            mse_list = []
            mape_list = []
            mae_list = []
            metrics = {}
            for tid, true_arr in aligned_actual.items():
                est_arr = contribs.get(tid, np.zeros(len(union_ts)))

                mse = np.mean((true_arr - est_arr) ** 2)

                mask = true_arr > 0
                if np.any(mask):
                    mape = np.mean(np.abs(true_arr[mask] - est_arr[mask]) / true_arr[mask]) * 100
                else:
                    mape = np.nan

                mae = np.mean(np.abs(true_arr - est_arr))

                benchmark_result[wname][utilization][tid] = {
                    "MSE": mse,
                    "MAPE": mape,
                    "MAE": mae
                }
                mse_list.append(mse)
                mape_list.append(mape)
                mae_list.append(mae)

            avg_mse = np.mean(mse_list)
            avg_mape = np.nanmean(mape_list)
            avg_mae = np.mean(mae_list)
            metrics["average_MSE"] = float(avg_mse)
            metrics["average_MAPE"] = float(avg_mape)
            metrics["average_MAE"] = float(avg_mae)

    return contribs_all, benchmark_result, metrics

def gm_benchmark(benchmark):
    contribs_all = {}
    benchmark_result = {}

    for wi, workload_entry in enumerate(benchmark):
        wname = workload_entry.get("workload_name", f"w{wi}")
        contribs_all[wname] = {}
        benchmark_result[wname] = {}

        for utilization in ["cpu", "memory"]:
            actual_task_contribs = {}
            tasks = []
            all_ts = []

            for job in workload_entry.get("jobs", []):
                job_id = int(job["job_id"])

                if utilization == "cpu":
                    pairs = job.get("cpu_util", [])
                else:
                    pairs = job.get("memory_util", [])

                if not pairs:
                    continue

                ts, vals = zip(*pairs)
                ts = np.array(list(map(float, ts)))
                vals = np.array(list(map(float, vals)))

                order = np.argsort(ts)
                ts = ts[order]
                vals = vals[order]

                actual_task_contribs[job_id] = {
                    "timestamps": ts,
                    "util": vals
                }

                start = float(job.get("start_time", ts[0]))
                finish = float(job.get("end_time", ts[-1]))

                tasks.append({
                    "task_id": job_id,
                    "start": start,
                    "finish": finish,
                })

                all_ts.extend(ts.tolist())

            if not tasks:
                continue

            union_ts = np.array(sorted(set(all_ts)), dtype=float)
            observed = np.zeros(len(union_ts))

            aligned_actual = {}
            for tid, data in actual_task_contribs.items():
                ts = data["timestamps"]
                vals = data["util"]

                arr = np.zeros(len(union_ts))
                ts_to_val = {float(t): float(v) for t, v in zip(ts, vals)}

                for i, t in enumerate(union_ts):
                    if t in ts_to_val:
                        arr[i] = ts_to_val[t]

                aligned_actual[tid] = arr
                observed += arr

            T = len(union_ts)
            Ntasks = len(tasks)

            active = np.zeros((Ntasks, T), dtype=bool)
            tids = []

            for j, tinfo in enumerate(tasks):
                tids.append(tinfo["task_id"])
                active[j] = (union_ts >= tinfo["start"]) & (union_ts < tinfo["finish"])

            residual = observed.copy()
            known = np.zeros(Ntasks, dtype=bool)
            mean_est = np.full(Ntasks, np.nan)
            contribs = np.zeros((Ntasks, T))

            for iters in range(40):
                progress = False

                for j in range(Ntasks):
                    if known[j]:
                        continue

                    mask_only = active[j]

                    if not mask_only.any():
                        mask_ok = active[j].copy()
                        for k in range(Ntasks):
                            if k == j or known[k]:
                                continue
                            mask_ok &= ~active[k]
                        mask_candidate = mask_ok
                    else:
                        mask_candidate = mask_only

                    if mask_candidate.any():
                        est = float(np.mean(residual[mask_candidate]))
                        est = max(0.0, est)

                        mean_est[j] = est
                        contribs[j] = active[j].astype(float) * est

                        residual -= contribs[j]
                        known[j] = True
                        progress = True

                if not progress:
                    break

            active_contribs = []
            result_this_util = {}

            for j in range(Ntasks):
                tid = tids[j]
                util_arr = active[j].astype(float) * mean_est[j]

                mask = util_arr > 0.0
                active_contribs.append({
                    "task_id": tid,
                    "Util": list(zip(union_ts[mask].tolist(), util_arr[mask].tolist()))
                })

                result_this_util[tid] = {
                    "timestamps": union_ts,
                    "reconstructed": util_arr,
                    "actual": aligned_actual.get(tid, np.zeros(T))
                }

            contribs_all[wname][utilization] = active_contribs
            benchmark_result[wname][utilization] = result_this_util

    def compute_metrics(true, est):
        mask = true > 0  # avoid division by zero for MAPE

        mae = np.mean(np.abs(est - true))
        mse = np.mean((est - true) ** 2)

        if np.any(mask):
            mape = np.mean(np.abs((est[mask] - true[mask]) / true[mask])) * 100
        else:
            mape = np.nan

        return mae, mse, mape

    metrics_all = {}
    mae_list = []
    mape_list = []
    mse_list = []
    for wname in benchmark_result:

        for utilization in benchmark_result[wname]:


            for tid, data in benchmark_result[wname][utilization].items():
                actual_data = data["actual"]
                est = data["reconstructed"]

                mae, mse, mape = compute_metrics(actual_data, est)
                mae_list.append(mae)
                mse_list.append(mse)
                mape_list.append(mape)
    
    avg_mae = np.mean(mae_list)
    avg_mse = np.mean(mse_list)
    avg_mape = np.nanmean(mape_list)
    metrics_all["average_MSE"] = float(avg_mse)
    metrics_all["average_MAPE"] = float(avg_mape)
    metrics_all["average_MAE"] = float(avg_mae)

    return contribs_all, benchmark_result, metrics_all

if __name__ == "__main__":
    with open("benchmark_new.json", "r") as f:
        benchmark_data = json.load(f)
    contribs_nnls, results_nnls, metrics_nnls = NNLS_benchmark(benchmark_data)
    print("NNLS Metrics:", metrics_nnls)
    contribs_gm, results_gm, metrics_gm = gm_benchmark(benchmark_data)
    print("GM Metrics:", metrics_gm)