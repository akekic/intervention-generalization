from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from intervention_generalization.data_generator import PolynomialSCM, sample_polynomial_scm_params


def calculate_summary_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
        "mean": np.mean(data, axis=0),
        "std": np.std(data, axis=0)
    }


def print_summary_stats(stats: Dict[str, Dict[str, np.ndarray]], name: str):
    print(f"\n{name} Summary Statistics:")
    for var_idx in range(len(stats["observational"]["mean"])):
        print(f"  Variable {var_idx}:")
        for sample_type in stats.keys():
            print(f"    {sample_type.capitalize()}:")
            print(f"      Range: ({stats[sample_type]['min'][var_idx]:.4f}, {stats[sample_type]['max'][var_idx]:.4f})")
            print(f"      Mean:  {stats[sample_type]['mean'][var_idx]:.4f}")
            print(f"      Std:   {stats[sample_type]['std'][var_idx]:.4f}")


def print_distribution_types(scm: PolynomialSCM):
    print("\nDistribution Types:")
    print("  W (Exogenous noise for C):")
    for k, params in enumerate(scm.W_params):
        print(f"    Variable {k}: {params.dist_type.value}")

    print("  V (Exogenous noise for A):")
    for k, params in enumerate(scm.V_params):
        print(f"    Variable {k}: {params.dist_type.value}")

    print(f"  U (Exogenous noise for Y): {scm.U_params.dist_type.value}")


def generate_all_data(scm: PolynomialSCM, N: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    A_obs, Y_obs = scm.generate_obs(N)

    A_data = {"observational": A_obs}
    Y_data = {"observational": Y_obs}

    # Generate single interventional data for all intervention variables
    for k in range(scm.K):
        A_single_int, Y_single_int = scm.generate_single_int(intervention_var=k, N=N)
        A_data[f"single_intervention_{k}"] = A_single_int
        Y_data[f"single_intervention_{k}"] = Y_single_int

    A_joint_int, Y_joint_int = scm.generate_joint_int(N=N)
    A_data["joint_intervention"] = A_joint_int
    Y_data["joint_intervention"] = Y_joint_int

    return A_data, Y_data


def plot_summary_stats(A_stats: Dict[str, Dict[str, np.ndarray]], Y_stats: Dict[str, Dict[str, np.ndarray]], K: int):
    fig, axs = plt.subplots(K + 1, 1, figsize=(10, 5 * (K + 1)), sharex=True)
    fig.suptitle("Summary Statistics for Actions (A) and Outcome (Y)")

    sample_types = list(A_stats.keys())
    x = range(len(sample_types))

    for k in range(K):
        means = [A_stats[st]["mean"][k] for st in sample_types]
        stds = [A_stats[st]["std"][k] for st in sample_types]
        axs[k].errorbar(x, means, yerr=stds, fmt='o', capsize=5)
        axs[k].set_ylabel(f"Action A[{k}]")
        axs[k].set_title(f"Action Variable A[{k}]")

    means = [Y_stats[st]["mean"][0] for st in sample_types]
    stds = [Y_stats[st]["std"][0] for st in sample_types]
    axs[-1].errorbar(x, means, yerr=stds, fmt='o', capsize=5)
    axs[-1].set_ylabel("Outcome Y")
    axs[-1].set_title("Outcome Variable Y")

    plt.xticks(x, sample_types, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(58)  # For reproducibility
    K = 5  # Number of action variables
    N = 1000  # Number of samples

    # Sample parameters and create PolynomialSCM instance
    params = sample_polynomial_scm_params(K)
    scm = PolynomialSCM(**params.__dict__)

    print("Polynomial SCM Parameters:")
    print(f"  K: {scm.K}")
    print(f"  p_edge: {scm.p_edge}")
    print(f"  sigma_int: {scm.sigma_int}")
    print("  Delta mask:")
    print(scm.delta_mask)

    # Print distribution types
    print_distribution_types(scm)

    # Generate all types of data
    A_data, Y_data = generate_all_data(scm, N)

    # Calculate summary statistics
    A_stats = {sample_type: calculate_summary_stats(data) for sample_type, data in A_data.items()}
    Y_stats = {sample_type: calculate_summary_stats(data.reshape(-1, 1)) for sample_type, data in Y_data.items()}

    # Print summary statistics
    print_summary_stats(A_stats, "Actions (A)")
    print_summary_stats(Y_stats, "Outcome (Y)")

    # Plot summary statistics
    plot_summary_stats(A_stats, Y_stats, K)


if __name__ == "__main__":
    main()