from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_data(
    A_obs: np.ndarray,
    Y_obs: np.ndarray,
    A_sint: Iterable[np.ndarray],
    Y_sint: Iterable[np.ndarray],
    A_jint: np.ndarray,
    Y_jint: np.ndarray,
    A_pool: np.ndarray,
    Y_pool: np.ndarray,
) -> plt.Figure:
    K = A_obs.shape[1]  # Number of action variables
    # row of K+1 figures
    fig, axs = plt.subplots(K + 1, 1, figsize=(6, 4 * (K + 1)))
    # plot a sns distplot for each variable in different data sets
    for k in range(K):
        sns.distplot(A_obs[:, k], ax=axs[k], label="obs")
        for j, A_k_sint in enumerate(A_sint):
            sns.distplot(A_k_sint[:, k], ax=axs[k], label=f"sint_{j}")
        sns.distplot(A_jint[:, k], ax=axs[k], label="jint")
        sns.distplot(A_pool[:, k], ax=axs[k], label="pool")
        axs[k].set_title(f"A_{k}")
        axs[k].legend()

    # plot a sns distplot for Y in different data sets
    sns.distplot(Y_obs, ax=axs[K], label="obs")
    for j, Y_k_sint in enumerate(Y_sint):
        sns.distplot(Y_k_sint, ax=axs[K], label=f"sint_{j}")
    sns.distplot(Y_jint, ax=axs[K], label="jint")
    sns.distplot(Y_pool, ax=axs[K], label="pool")
    axs[K].set_title("Y")
    axs[K].legend()
    plt.tight_layout()

    return fig
