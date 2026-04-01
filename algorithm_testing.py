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

    mae_list = []
    mse_list = []
    mape_list = []

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
                active_sum = active.sum(axis=0)

                for j in range(Ntasks):
                    if known[j]:
                        continue

                    mask_only = active[j] & (active_sum == 1)

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

                if np.isnan(mean_est[j]):
                    util_arr = np.zeros(T)
                else:
                    util_arr = active[j].astype(float) * mean_est[j]

                true_arr = aligned_actual.get(tid, np.zeros(T))

                mse = np.mean((true_arr - util_arr) ** 2)
                mae = np.mean(np.abs(true_arr - util_arr))

                mask = true_arr > 0
                if np.any(mask):
                    mape = np.mean(np.abs(true_arr[mask] - util_arr[mask]) / true_arr[mask]) * 100
                else:
                    mape = np.nan

                mse_list.append(mse)
                mae_list.append(mae)
                mape_list.append(mape)

                mask_util = util_arr > 0.0
                active_contribs.append({
                    "task_id": tid,
                    "Util": list(zip(union_ts[mask_util].tolist(), util_arr[mask_util].tolist()))
                })

                result_this_util[tid] = {
                    "timestamps": union_ts.tolist(),
                    "reconstructed": util_arr.tolist(),
                    "actual": true_arr.tolist(),
                    "MSE": float(mse),
                    "MAPE": float(mape) if not np.isnan(mape) else np.nan,
                    "MAE": float(mae)
                }

            contribs_all[wname][utilization] = active_contribs
            benchmark_result[wname][utilization] = result_this_util

    metrics_all = {
        "average_MSE": float(np.mean(mse_list)) if mse_list else np.nan,
        "average_MAPE": float(np.nanmean(mape_list)) if mape_list else np.nan,
        "average_MAE": float(np.mean(mae_list)) if mae_list else np.nan
    }

    return contribs_all, benchmark_result, metrics_all

if __name__ == "__main__":
    with open("benchmark_new.json", "r") as f:
        benchmark_data = json.load(f)
    contribs_nnls, results_nnls, metrics_nnls = NNLS_benchmark(benchmark_data)
    print("NNLS Metrics:", metrics_nnls)
    contribs_gm, results_gm, metrics_gm = gm_benchmark(benchmark_data)
    print("GM Metrics:", metrics_gm)