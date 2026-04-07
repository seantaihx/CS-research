import json
import numpy as np
import matplotlib.pyplot as plt
from algorithm_blockwise_nonnegative_least_squares import mean_then_blockwise_nnls


def _has_nan(actual, reconstructed):
    actual = np.asarray(actual, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    return np.isnan(actual).any() or np.isnan(reconstructed).any()


def _calculate_job_metrics(actual, reconstructed):
    actual = np.asarray(actual, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)

    mae = np.mean(np.abs(actual - reconstructed))
    mse = np.mean((actual - reconstructed) ** 2)

    mask_active = actual > 0
    if np.any(mask_active):
        mape = np.mean(
            np.abs((reconstructed[mask_active] - actual[mask_active]) / actual[mask_active])
        ) * 100
    else:
        mape = np.nan

    return {
        "MAE_all": mae,
        "MSE_all": mse,
        "MAPE_all": mape
    }


def _finalize_metrics(mse_all_list, mape_all_list, mae_all_list):
    return {
        "average_MSE_all": float(np.nanmean(mse_all_list)) if mse_all_list else np.nan,
        "average_MAPE_all": float(np.nanmean(mape_all_list)) if mape_all_list else np.nan,
        "average_MAE_all": float(np.nanmean(mae_all_list)) if mae_all_list else np.nan,
    }


def _prepare_workload_data(workload_entry, utilization):
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
        return None, None, None, None

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

    return tasks, union_ts, observed, aligned_actual


def gm_benchmark(benchmark):
    contribs_all = {}
    benchmark_result = {}

    gm_fail_count = 0
    gm_total_tasks = 0

    for wi, workload_entry in enumerate(benchmark):
        wname = workload_entry.get("workload_name", f"w{wi}")
        contribs_all[wname] = {}
        benchmark_result[wname] = {}

        for utilization in ["cpu", "memory"]:
            tasks, union_ts, observed, aligned_actual = _prepare_workload_data(workload_entry, utilization)

            if tasks is None:
                continue

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

            for _ in range(40):
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
                        est = float(np.nanmean(residual[mask_candidate]))
                        est = max(0.0, est)

                        mean_est[j] = est
                        contribs[j] = active[j].astype(float) * est

                        residual -= contribs[j]
                        known[j] = True
                        progress = True

                if not progress:
                    break

            active_contribs = []
            benchmark_result[wname][utilization] = {}

            for j in range(Ntasks):
                tid = tids[j]
                gm_total_tasks += 1

                if np.isnan(mean_est[j]):
                    util_arr = np.full(T, np.nan)
                    failed = True
                    gm_fail_count += 1
                else:
                    util_arr = active[j].astype(float) * mean_est[j]
                    failed = False

                mask_util = np.isfinite(util_arr) & (util_arr > 0.0)
                active_contribs.append({
                    "task_id": tid,
                    "Util": list(zip(union_ts[mask_util].tolist(), util_arr[mask_util].tolist()))
                })

                true_arr = aligned_actual.get(tid, np.zeros(T))

                benchmark_result[wname][utilization][tid] = {
                    "timestamps": union_ts,
                    "actual": true_arr,
                    "reconstructed": util_arr,
                    "failed": failed,
                    "has_nan": _has_nan(true_arr, util_arr)
                }

            contribs_all[wname][utilization] = active_contribs

    print(f"GM failed to estimate {gm_fail_count} out of {gm_total_tasks} tasks.")
    return contribs_all, benchmark_result


def NNLS_benchmark(benchmark, block_size=30, l2=1e-2):
    contribs_all = {}
    benchmark_result = {}

    for wi, workload_entry in enumerate(benchmark):
        wname = workload_entry.get("workload_name", f"w{wi}")
        contribs_all[wname] = {}
        benchmark_result[wname] = {}

        for utilization in ["cpu", "memory"]:
            tasks, union_ts, observed, aligned_actual = _prepare_workload_data(workload_entry, utilization)

            if tasks is None:
                continue

            contribs = mean_then_blockwise_nnls(union_ts, observed, tasks, block_size, l2)

            contribs_all[wname][utilization] = contribs
            benchmark_result[wname][utilization] = {}

            for tid, true_arr in aligned_actual.items():
                est_arr = contribs.get(tid, np.zeros(len(union_ts)))

                benchmark_result[wname][utilization][tid] = {
                    "timestamps": union_ts,
                    "actual": true_arr,
                    "reconstructed": est_arr,
                    "failed": False,
                    "has_nan": _has_nan(true_arr, est_arr)
                }

    return contribs_all, benchmark_result



def calculate_metrics(benchmark_result, skip_failed=True, skip_nan=True):
    mse_all_list = []
    mape_all_list = []
    mae_all_list = []

    valid_jobs = {}

    for wname in benchmark_result:
        valid_jobs[wname] = {}

        for utilization in benchmark_result[wname]:
            valid_jobs[wname][utilization] = {}

            for tid, data in benchmark_result[wname][utilization].items():
                failed = data.get("failed", False)
                has_nan = data.get("has_nan", False)

                if skip_failed and failed:
                    valid_jobs[wname][utilization][tid] = False
                    continue

                if skip_nan and has_nan:
                    valid_jobs[wname][utilization][tid] = False
                    continue

                metrics = _calculate_job_metrics(data["actual"], data["reconstructed"])

                data["MSE_all"] = metrics["MSE_all"]
                data["MAPE_all"] = metrics["MAPE_all"]
                data["MAE_all"] = metrics["MAE_all"]

                mse_all_list.append(metrics["MSE_all"])
                mae_all_list.append(metrics["MAE_all"])
                mape_all_list.append(metrics["MAPE_all"])

                valid_jobs[wname][utilization][tid] = True

    metrics_all = _finalize_metrics(mse_all_list, mape_all_list, mae_all_list)
    return metrics_all, valid_jobs


def calculate_metrics_with_mask(benchmark_result, valid_jobs_mask):
    mse_all_list = []
    mape_all_list = []
    mae_all_list = []

    for wname in benchmark_result:
        for utilization in benchmark_result[wname]:
            for tid, data in benchmark_result[wname][utilization].items():
                is_valid = valid_jobs_mask.get(wname, {}).get(utilization, {}).get(tid, False)
                if not is_valid:
                    continue

                metrics = _calculate_job_metrics(data["actual"], data["reconstructed"])

                data["MSE_all"] = metrics["MSE_all"]
                data["MAPE_all"] = metrics["MAPE_all"]
                data["MAE_all"] = metrics["MAE_all"]

                mse_all_list.append(metrics["MSE_all"])
                mae_all_list.append(metrics["MAE_all"])
                mape_all_list.append(metrics["MAPE_all"])

    metrics_all = _finalize_metrics(mse_all_list, mape_all_list, mae_all_list)
    return metrics_all



def _to_1d_clean(values):
    arr = np.asarray(values, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def _cdf_xy(values):
    x = np.sort(_to_1d_clean(values))
    if len(x) == 0:
        return np.array([]), np.array([])
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def _percentile(values, p):
    v = _to_1d_clean(values)
    if v.size == 0:
        return np.nan
    return float(np.percentile(v, p))


def _cdf_y_at_x(values, x_target):
    x, y = _cdf_xy(values)
    if len(x) == 0:
        return np.nan

    idx = np.searchsorted(x, x_target, side="right")
    if idx == 0:
        return 0.0
    return float(y[idx - 1])

def plot_benchmark_cdf(results_nnls, results_gm, save_path="benchmark_cdf.png", p_tail=95):
    actual_cpu = []
    actual_mem = []

    nnls_cpu = []
    nnls_mem = []

    gm_cpu = []
    gm_mem = []

    for wname in results_nnls:
        for utilization in results_nnls[wname]:
            tids = list(results_nnls[wname][utilization].keys())
            if not tids:
                continue

            first_tid = tids[0]
            T = len(results_nnls[wname][utilization][first_tid]["actual"])

            actual_sum = np.zeros(T)
            nnls_sum = np.zeros(T)

            for tid in tids:
                data = results_nnls[wname][utilization][tid]
                actual_sum += np.asarray(data["actual"], dtype=float)
                nnls_sum += np.asarray(data["reconstructed"], dtype=float)

            if utilization == "cpu":
                actual_cpu.extend(actual_sum.tolist())
                nnls_cpu.extend(nnls_sum.tolist())
            elif utilization == "memory":
                actual_mem.extend(actual_sum.tolist())
                nnls_mem.extend(nnls_sum.tolist())


    for wname in results_gm:
        for utilization in results_gm[wname]:
            tids = list(results_gm[wname][utilization].keys())
            if not tids:
                continue

            first_tid = tids[0]
            T = len(results_gm[wname][utilization][first_tid]["actual"])

            gm_sum = np.zeros(T)
            valid_mask = np.zeros(T, dtype=bool)

            for tid in tids:
                data = results_gm[wname][utilization][tid]
                arr = np.asarray(data["reconstructed"], dtype=float)

                not_nan = ~np.isnan(arr)
                gm_sum[not_nan] += arr[not_nan]
                valid_mask |= not_nan

            gm_sum = gm_sum[valid_mask]

            if utilization == "cpu":
                gm_cpu.extend(gm_sum.tolist())
            elif utilization == "memory":
                gm_mem.extend(gm_sum.tolist())

    print("Finished collecting data for CDF plot.")

    x_actual_cpu, y_actual_cpu = _cdf_xy(actual_cpu)
    x_nnls_cpu, y_nnls_cpu = _cdf_xy(nnls_cpu)
    x_gm_cpu, y_gm_cpu = _cdf_xy(gm_cpu)

    x_actual_mem, y_actual_mem = _cdf_xy(actual_mem)
    x_nnls_mem, y_nnls_mem = _cdf_xy(nnls_mem)
    x_gm_mem, y_gm_mem = _cdf_xy(gm_mem)

    print("Start creating plot.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ================= CPU =================
    axes[0].plot(x_actual_cpu, y_actual_cpu, label="Original CPU")
    axes[0].plot(x_nnls_cpu, y_nnls_cpu, label="NNLS Reconstructed CPU")
    axes[0].plot(x_gm_cpu, y_gm_cpu, label="GM Reconstructed CPU")

    # mark x = 20 and 60
    for x_mark in [20, 60]:
        axes[0].axvline(x_mark, linestyle="--", linewidth=1, alpha=0.7)

        y_actual = _cdf_y_at_x(actual_cpu, x_mark)
        y_nnls = _cdf_y_at_x(nnls_cpu, x_mark)
        y_gm = _cdf_y_at_x(gm_cpu, x_mark)

        if np.isfinite(y_actual):
            axes[0].hlines(y_actual, xmin=0, xmax=x_mark, linestyles=":", linewidth=1, alpha=0.7)
            axes[0].plot(x_mark, y_actual, marker="o")
            axes[0].annotate(f"({x_mark}, {y_actual:.2f})", (x_mark, y_actual),
                             textcoords="offset points", xytext=(5, 5), fontsize=8)

        if np.isfinite(y_nnls):
            axes[0].hlines(y_nnls, xmin=0, xmax=x_mark, linestyles=":", linewidth=1, alpha=0.7)
            axes[0].plot(x_mark, y_nnls, marker="o")
            axes[0].annotate(f"({x_mark}, {y_nnls:.2f})", (x_mark, y_nnls),
                             textcoords="offset points", xytext=(5, -10), fontsize=8)

        if np.isfinite(y_gm):
            axes[0].hlines(y_gm, xmin=0, xmax=x_mark, linestyles=":", linewidth=1, alpha=0.7)
            axes[0].plot(x_mark, y_gm, marker="o")
            axes[0].annotate(f"({x_mark}, {y_gm:.2f})", (x_mark, y_gm),
                             textcoords="offset points", xytext=(5, 15), fontsize=8)

    # percentiles
    cpu_actual_p50 = _percentile(actual_cpu, 50)
    cpu_actual_p95 = _percentile(actual_cpu, p_tail)
    cpu_nnls_p50 = _percentile(nnls_cpu, 50)
    cpu_nnls_p95 = _percentile(nnls_cpu, p_tail)
    cpu_gm_p50 = _percentile(gm_cpu, 50)
    cpu_gm_p95 = _percentile(gm_cpu, p_tail)

    for p in [cpu_actual_p50, cpu_nnls_p50, cpu_gm_p50]:
        if np.isfinite(p):
            axes[0].axvline(p, linestyle="--", linewidth=1)

    for p in [cpu_actual_p95, cpu_nnls_p95, cpu_gm_p95]:
        if np.isfinite(p):
            axes[0].axvline(p, linestyle=":", linewidth=1)

    axes[0].set_title("CDF of CPU Utilization")
    axes[0].set_xlabel("CPU Utilization")
    axes[0].set_ylabel("CDF")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ================= Memory =================
    axes[1].plot(x_actual_mem, y_actual_mem, label="Original Memory")
    axes[1].plot(x_nnls_mem, y_nnls_mem, label="NNLS Reconstructed Memory")
    axes[1].plot(x_gm_mem, y_gm_mem, label="GM Reconstructed Memory")

    for x_mark in [20, 60]:
        axes[1].axvline(x_mark, linestyle="--", linewidth=1, alpha=0.7)

        y_actual = _cdf_y_at_x(actual_mem, x_mark)
        y_nnls = _cdf_y_at_x(nnls_mem, x_mark)
        y_gm = _cdf_y_at_x(gm_mem, x_mark)

        if np.isfinite(y_actual):
            axes[1].hlines(y_actual, xmin=0, xmax=x_mark, linestyles=":", linewidth=1, alpha=0.7)
            axes[1].plot(x_mark, y_actual, marker="o")
            axes[1].annotate(f"({x_mark}, {y_actual:.2f})", (x_mark, y_actual),
                             textcoords="offset points", xytext=(5, 5), fontsize=8)

        if np.isfinite(y_nnls):
            axes[1].hlines(y_nnls, xmin=0, xmax=x_mark, linestyles=":", linewidth=1, alpha=0.7)
            axes[1].plot(x_mark, y_nnls, marker="o")
            axes[1].annotate(f"({x_mark}, {y_nnls:.2f})", (x_mark, y_nnls),
                             textcoords="offset points", xytext=(5, -10), fontsize=8)

        if np.isfinite(y_gm):
            axes[1].hlines(y_gm, xmin=0, xmax=x_mark, linestyles=":", linewidth=1, alpha=0.7)
            axes[1].plot(x_mark, y_gm, marker="o")
            axes[1].annotate(f"({x_mark}, {y_gm:.2f})", (x_mark, y_gm),
                             textcoords="offset points", xytext=(5, 15), fontsize=8)

    mem_actual_p50 = _percentile(actual_mem, 50)
    mem_actual_p95 = _percentile(actual_mem, p_tail)
    mem_nnls_p50 = _percentile(nnls_mem, 50)
    mem_nnls_p95 = _percentile(nnls_mem, p_tail)
    mem_gm_p50 = _percentile(gm_mem, 50)
    mem_gm_p95 = _percentile(gm_mem, p_tail)

    for p in [mem_actual_p50, mem_nnls_p50, mem_gm_p50]:
        if np.isfinite(p):
            axes[1].axvline(p, linestyle="--", linewidth=1)

    for p in [mem_actual_p95, mem_nnls_p95, mem_gm_p95]:
        if np.isfinite(p):
            axes[1].axvline(p, linestyle=":", linewidth=1)

    axes[1].set_title("CDF of Memory Utilization")
    axes[1].set_xlabel("Memory Utilization")
    axes[1].set_ylabel("CDF")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    print("Finished plotting.")

    print("\nCPU Utilization percentiles")
    print(f"  Original: p50={cpu_actual_p50:.6g}, p{p_tail}={cpu_actual_p95:.6g}")
    print(f"  NNLS:     p50={cpu_nnls_p50:.6g}, p{p_tail}={cpu_nnls_p95:.6g}")
    print(f"  GM:       p50={cpu_gm_p50:.6g}, p{p_tail}={cpu_gm_p95:.6g}")

    print("\nMemory Utilization percentiles")
    print(f"  Original: p50={mem_actual_p50:.6g}, p{p_tail}={mem_actual_p95:.6g}")
    print(f"  NNLS:     p50={mem_nnls_p50:.6g}, p{p_tail}={mem_nnls_p95:.6g}")
    print(f"  GM:       p50={mem_gm_p50:.6g}, p{p_tail}={mem_gm_p95:.6g}")

    print("\nCDF values at x = 20 and x = 60")
    for x_mark in [20, 60]:
        print(f"\nCPU at x={x_mark}")
        print(f"  Original y={_cdf_y_at_x(actual_cpu, x_mark):.6g}")
        print(f"  NNLS     y={_cdf_y_at_x(nnls_cpu, x_mark):.6g}")
        print(f"  GM       y={_cdf_y_at_x(gm_cpu, x_mark):.6g}")

        print(f"Memory at x={x_mark}")
        print(f"  Original y={_cdf_y_at_x(actual_mem, x_mark):.6g}")
        print(f"  NNLS     y={_cdf_y_at_x(nnls_mem, x_mark):.6g}")
        print(f"  GM       y={_cdf_y_at_x(gm_mem, x_mark):.6g}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "cpu": {
            "original_p50": cpu_actual_p50,
            f"original_p{p_tail}": cpu_actual_p95,
            "nnls_p50": cpu_nnls_p50,
            f"nnls_p{p_tail}": cpu_nnls_p95,
            "gm_p50": cpu_gm_p50,
            f"gm_p{p_tail}": cpu_gm_p95,
            "cdf_at_20": {
                "original": _cdf_y_at_x(actual_cpu, 20),
                "nnls": _cdf_y_at_x(nnls_cpu, 20),
                "gm": _cdf_y_at_x(gm_cpu, 20),
            },
            "cdf_at_60": {
                "original": _cdf_y_at_x(actual_cpu, 60),
                "nnls": _cdf_y_at_x(nnls_cpu, 60),
                "gm": _cdf_y_at_x(gm_cpu, 60),
            },
        },
        "memory": {
            "original_p50": mem_actual_p50,
            f"original_p{p_tail}": mem_actual_p95,
            "nnls_p50": mem_nnls_p50,
            f"nnls_p{p_tail}": mem_nnls_p95,
            "gm_p50": mem_gm_p50,
            f"gm_p{p_tail}": mem_gm_p95,
            "cdf_at_20": {
                "original": _cdf_y_at_x(actual_mem, 20),
                "nnls": _cdf_y_at_x(nnls_mem, 20),
                "gm": _cdf_y_at_x(gm_mem, 20),
            },
            "cdf_at_60": {
                "original": _cdf_y_at_x(actual_mem, 60),
                "nnls": _cdf_y_at_x(nnls_mem, 60),
                "gm": _cdf_y_at_x(gm_mem, 60),
            },
        }
    }

if __name__ == "__main__":
    with open("benchmark_new5.json", "r") as f:
        benchmark_data = json.load(f)

    contribs_gm, results_gm = gm_benchmark(benchmark_data)
    contribs_nnls, results_nnls = NNLS_benchmark(benchmark_data)

    metrics_gm, valid_jobs_gm = calculate_metrics(results_gm)
    print("GM Metrics:", metrics_gm)

    metrics_nnls = calculate_metrics_with_mask(results_nnls, valid_jobs_gm)
    print("NNLS Metrics:", metrics_nnls)

    cdf_stats = plot_benchmark_cdf(
        results_nnls,
        results_gm,
        save_path="cdf_original_vs_reconstructed5.png",
        p_tail=95
    )

    print("\nReturned CDF stats:")
    print(cdf_stats)