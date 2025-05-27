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


def run_single_experiment(args_and_seed, n_jobs=None, verbose=False):
    args, seed = args_and_seed
    np.random.seed(seed)

    if args.scm_type == "linear":
        scm = sample_linear_scm(args.K)
    else:
        scm = sample_polynomial_scm(args.K)
    return run_experiment(
        scm=scm,
        K=args.K,
        N=args.N,
        n_order_fit=args.n_order_fit,
        regularization=args.regularization,
        n_jobs=n_jobs,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run SCM experiments with customizable parameters."
    )
    parser.add_argument(
        "--N", type=int, default=10000, help="Number of samples (default: 10000)"
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
        help="Number of experiment runs (default: 100)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for single experiment run (default: 42)",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_subdir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)

    # Create a CSV file to store RMSE values
    csv_path = os.path.abspath(os.path.join(output_subdir, "rmse_values.csv"))
    with open(csv_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            ["Run", "RMSE Obs", "RMSE Method", "RMSE Joint Int", "RMSE Pool"]
        )

        if args.N_runs == 1:
            results = [
                run_single_experiment(
                    (args, args.seed), n_jobs=mp.cpu_count(), verbose=True
                )
            ]
        else:
            pool = mp.Pool(processes=mp.cpu_count())
            results = list(
                tqdm(
                    pool.imap(
                        run_single_experiment, [(args, s) for s in range(args.N_runs)]
                    ),
                    total=args.N_runs,
                )
            )
            pool.close()
            pool.join()

        # Unpack results
        rmse_obs = np.array([res["rmse_obs"] for res in results])
        rmse_method = np.array([res["rmse_method"] for res in results])
        rmse_jint = np.array([res["rmse_jint"] for res in results])
        rmse_pool = np.array([res["rmse_pool"] for res in results])

        if args.N_runs == 1:
            rmse_obs_mean, rmse_obs_sem = rmse_obs[0], 0
            rmse_method_mean, rmse_method_sem = rmse_method[0], 0
            rmse_jint_mean, rmse_jint_sem = rmse_jint[0], 0
            rmse_pool_mean, rmse_pool_sem = rmse_pool[0], 0
        else:
            rmse_obs_mean, rmse_obs_sem = calc_stats(rmse_obs)
            rmse_method_mean, rmse_method_sem = calc_stats(rmse_method)
            rmse_jint_mean, rmse_jint_sem = calc_stats(rmse_jint)
            rmse_pool_mean, rmse_pool_sem = calc_stats(rmse_pool)

        print(f"RMSE using obs model for jint: {rmse_obs_mean} +- {rmse_obs_sem}")
        print(f"RMSE using our method: {rmse_method_mean} +- {rmse_method_sem}")
        print(f"RMSE training on jint (topline): {rmse_jint_mean} +- {rmse_jint_sem}")
        print(
            f"RMSE training on pooled (obs, sint): {rmse_pool_mean} +- {rmse_pool_sem}"
        )

        # Write individual RMSE values to CSV
        for run in range(args.N_runs):
            csvwriter.writerow(
                [run, rmse_obs[run], rmse_method[run], rmse_jint[run], rmse_pool[run]]
            )

    means = [rmse_method_mean, rmse_jint_mean, rmse_pool_mean, rmse_obs_mean]
    errors = [rmse_method_sem, rmse_jint_sem, rmse_pool_sem, rmse_obs_sem]
    labels = [
        "our method",
        "joint\nintervention\ndata",
        "pooled\ndata",
        "only\nobservational\ndata",
    ]

    plt.figure(figsize=(0.8 * SINGLE_COLUMN, 0.5 * SINGLE_COLUMN))
    if args.N_runs == 1:
        plt.scatter(range(1, len(means) + 1), means)
    else:
        plt.errorbar(
            range(1, len(means) + 1), means, yerr=errors, fmt="o", capsize=5, capthick=2
        )
    plt.ylabel("Mean RMSE")
    plt.xticks(range(1, len(means) + 1), labels)
    plt.tight_layout()

    plot_path = os.path.join(output_subdir, "experiment_plot.pdf")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

    # Save experiment parameters
    params_path = os.path.join(output_subdir, "experiment_params.txt")
    with open(params_path, "w") as f:
        f.write(f"N: {args.N}\n")
        f.write(f"K: {args.K}\n")
        f.write(f"n_order_fit: {args.n_order_fit}\n")
        f.write(f"N_runs: {args.N_runs}\n")
        f.write(f"SCM type: {args.scm_type}\n")
        f.write(f"Regularization: {args.regularization}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"JOB_NUM: {os.environ.get('JOB_NUM', -1)}\n")
        f.write(f"Commit hash: {os.popen('git rev-parse HEAD').read().strip()}\n")
    print(f"Experiment parameters saved to: {params_path}")


if __name__ == "__main__":
    main()
