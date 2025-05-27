import argparse
import csv
import multiprocessing as mp
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from intervention_generalization.data_generator import (
    sample_linear_scm,
    sample_polynomial_scm,
)
from intervention_generalization.utils import (
    calc_stats,
    SINGLE_COLUMN,
    style_modifications,
)
from intervention_generalization.utils import run_experiment

plt.rcParams.update(style_modifications)


def run_single_experiment(args_and_seed_and_ratio):
    args, seed, ratio = args_and_seed_and_ratio
    np.random.seed(seed)

    if args.scm_type == "linear":
        scm = sample_linear_scm(args.K)
    else:
        scm = sample_polynomial_scm(args.K)

    # compute data set sizes
    N_obs = int((args.K + 1) * args.N_avg / (args.K * ratio + 1))
    N_sint = int(ratio * N_obs)
    N_jint = args.N_avg
    N = [N_obs, N_sint, N_jint]
    return run_experiment(
        scm=scm,
        K=args.K,
        N=N,
        n_order_fit=args.n_order_fit,
        regularization=args.regularization,
        skip_pool=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with varying ratios of single-interventional and observational data."
    )
    parser.add_argument(
        "--N_avg",
        type=int,
        default=10000,
        help="Average number of samples per intervention setting (default: 10000)",
    )
    parser.add_argument(
        "--K", type=int, default=5, help="Number of single interventions (default: 5)"
    )
    parser.add_argument(
        "--n_order_fit",
        type=int,
        default=3,
        help="Order of polynomial fit (default: 3)",
    )
    parser.add_argument(
        "--N_runs",
        type=int,
        default=100,
        help="Number of experiment runs per data ratio setting (default: 100)",
    )
    parser.add_argument(
        "--scm_type",
        choices=["linear", "polynomial"],
        default="polynomial",
        help="Type of SCM to use (default: polynomial)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the plot (default: current directory)",
    )
    parser.add_argument(
        "--regularization",
        choices=["ridge", "lasso"],
        default="ridge",
        help="Type of regularization to use (default: ridge)",
    )
    parser.add_argument(
        "--ratios",
        type=lambda s: [float(item) for item in s.split(",")],
        default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,2.0,5.0,10.0",
        help="Comma-separated list of ratios of single-interventional to observational data"
        " (default: 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,2.0,5.0,10.0)",
    )

    args = parser.parse_args()

    RATIOS = args.ratios
    rmse_obs = []
    rmse_method = []
    rmse_jint = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_subdir = os.path.join(args.output_dir, f"data_ratio_exp_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)

    # Create a CSV file to store RMSE values
    csv_path = os.path.abspath(os.path.join(output_subdir, "rmse_values.csv"))
    with open(csv_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            ["Ratio", "Run", "RMSE Obs", "RMSE Method", "RMSE Joint Int", "RMSE Pool"]
        )

        # Run experiments
        for ratio in RATIOS:
            print(f"Running experiments for ratio: {ratio}")

            pool = mp.Pool(processes=mp.cpu_count())
            results = list(
                tqdm(
                    pool.imap(
                        run_single_experiment,
                        [(args, s, ratio) for s in range(args.N_runs)],
                    ),
                    total=args.N_runs,
                )
            )
            pool.close()
            pool.join()

            # Unpack results
            rmse_obs_r = np.array([res["rmse_obs"] for res in results])
            rmse_method_r = np.array([res["rmse_method"] for res in results])
            rmse_jint_r = np.array([res["rmse_jint"] for res in results])

            rmse_obs_mean, rmse_obs_sem = calc_stats(rmse_obs_r)
            rmse_method_mean, rmse_method_sem = calc_stats(rmse_method_r)
            rmse_jint_mean, rmse_jint_sem = calc_stats(rmse_jint_r)

            rmse_obs.append(rmse_obs_mean)
            rmse_method.append(rmse_method_mean)
            rmse_jint.append(rmse_jint_mean)

            # Write individual RMSE values to CSV
            for run in range(args.N_runs):
                csvwriter.writerow(
                    [
                        ratio,
                        run,
                        rmse_obs_r[run],
                        rmse_method_r[run],
                        rmse_jint_r[run],
                    ]
                )

    print(f"RMSE values saved to: {csv_path}")

    plt.figure(figsize=(0.8 * SINGLE_COLUMN, 0.5 * SINGLE_COLUMN))

    plt.plot(RATIOS, rmse_obs, "-o", label="obs", markersize=4)
    plt.fill_between(
        RATIOS, rmse_obs - rmse_obs_sem, rmse_obs + rmse_obs_sem, alpha=0.2
    )

    plt.plot(RATIOS, rmse_method, "-o", label="ours", markersize=4)
    plt.fill_between(
        RATIOS, rmse_method - rmse_method_sem, rmse_method + rmse_method_sem, alpha=0.2
    )

    plt.plot(RATIOS, rmse_jint, "-o", label="jint", markersize=4)
    plt.fill_between(
        RATIOS, rmse_jint - rmse_jint_sem, rmse_jint + rmse_jint_sem, alpha=0.2
    )

    plt.legend(fontsize=5)
    plt.xlabel("Ratio of single-interventional to observational data")
    plt.ylabel("Average RMSE")
    plt.xscale("log")

    plot_path = os.path.abspath(os.path.join(output_subdir, "experiment_plot.pdf"))
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

    # Save experiment parameters
    params_path = os.path.abspath(os.path.join(output_subdir, "experiment_params.txt"))
    with open(params_path, "w") as f:
        f.write(f"N_avg: {args.N_avg}\n")
        f.write(f"K: {args.K}\n")
        f.write(f"n_order_fit: {args.n_order_fit}\n")
        f.write(f"N_runs: {args.N_runs}\n")
        f.write(f"Ratios: {RATIOS}\n")
        f.write(f"SCM type: {args.scm_type}\n")
        f.write(f"Regularization: {args.regularization}\n")
        f.write(f"JOB_NUM: {os.environ.get('JOB_NUM', -1)}\n")
        f.write(f"Commit hash: {os.popen('git rev-parse HEAD').read().strip()}\n")
    print(f"Experiment parameters saved to: {params_path}")


if __name__ == "__main__":
    main()
