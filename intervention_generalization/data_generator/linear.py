from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .scm import SCM


class LinearSCM(SCM):
    """
    Linear Structural Causal Model (SCM).

    There is one outcome variable Y and K action variables A_1, ..., A_K. Each action variable is a
    direct cause of Y (A_k -> Y). There are also K confounders C_1, ..., C_K, where each confounder
    is a direct cause of A_k (C_k -> A_k) and Y (C_k -> Y).

    The SCM is defined by the following structural equations:
        Y := sum_k alpha_k A_k + sum_k beta_k C_k + U
        A_k := gamma_k C_k + sum_{j < k} delta_{k, j} delta_mask_{k, j} A_j + V_k
        C_k := W_k
        U ~ N(0, sigma_U^2), V_k ~ N(0, sigma_V[k]^2), W_k ~ N(0, sigma_W[k]^2)

    The delta_mask_{k, j} is a binary mask that determines whether the inter-action dependency between A_k and A_j
    is active. The probability of an edge being active is p_edge.

    Parameters
    ----------
    K : int
        Number of action and confounder variables.
    sigma_W : list
        Standard deviations of the exogenous noise terms W_k.
    sigma_V : list
        Standard deviations of the exogenous noise terms V_k.
    sigma_U : float
        Standard deviation of the exogenous noise term U.
    alpha : list
        Coefficients of the effect of action variables A_k in structural equation of the outcome variable Y.
    beta : list
        Coefficients of the effect of confounder variables C_k in structural equation of the outcome variable Y.
    gamma : list
        Coefficients of the effect of confounder variables C_k in structural equation of the action variables A_k.
    delta : list
        Coefficients of the inter-action dependencies between action variables A_k and A_j.
    delta_mask : list
        Mask for active edges in the inter-action dependencies.
    sigma_int : float
        Standard deviation of the stochastic intervention applied to the action variables.
    p_edge : float
        Probability of an edge being active in the inter-action dependencies.
    """

    def __init__(
        self,
        K: int,
        sigma_W: Iterable[float],
        sigma_V: Iterable[float],
        sigma_U: float,
        alpha: Iterable[float],
        beta: Iterable[float],
        gamma: Iterable[float],
        delta: Iterable[Iterable[float]],
        delta_mask: Iterable[Iterable[float]],
        sigma_int: float,
        p_edge: float,
    ):
        super().__init__()
        self.K = K
        self.sigma_W = np.array(sigma_W)
        self.sigma_V = np.array(sigma_V)
        self.sigma_U = sigma_U
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)
        self.gamma = np.array(gamma)
        self.delta = np.array(delta)
        self.delta_mask = np.array(delta_mask)
        self.sigma_int = sigma_int
        self.p_edge = p_edge

    def generate_noise(self, N: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate independent exogenous noise terms from lognormal distributions.

        Parameters
        ----------
        N : int, optional
            Number of samples to generate, by default 100.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Generated noise terms (W, U, V) where W has shape (N, K), U has shape (N,),
            and V has shape (N, K). All noise terms follow lognormal distributions.
        """
        W = np.random.lognormal(0, sigma=self.sigma_W, size=(N, self.K))
        V = np.random.lognormal(0, sigma=self.sigma_V, size=(N, self.K))
        U = np.random.lognormal(0, self.sigma_U, N)

        return W, U, V

    def generate_obs(self, N: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        W, U, V = self.generate_noise(N)

        C = W
        A = np.zeros((N, self.K))
        for k in range(self.K):
            A[:, k] = (
                self.gamma[k] * C[:, k]
                + np.dot(A[:, :k], self.delta[k, :k] * self.delta_mask[k, :k])
                + V[:, k]
            )

        Y = (self.alpha * A + self.beta * C).sum(axis=1) + U
        return A, Y

    def generate_single_int(
        self, intervention_var: int = 0, N: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        W, U, V = self.generate_noise(N)
        intervention = np.random.lognormal(0, self.sigma_int, N)

        C = W
        A = np.zeros((N, self.K))
        for k in range(self.K):
            if k == intervention_var:
                A[:, k] = intervention
            else:
                A[:, k] = (
                    self.gamma[k] * C[:, k]
                    + np.dot(A[:, :k], self.delta[k, :k] * self.delta_mask[k, :k])
                    + V[:, k]
                )

        Y = (self.alpha * A + self.beta * C).sum(axis=1) + U
        return A, Y

    def generate_joint_int(self, N: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        W, U, V = self.generate_noise(N)

        intervention = np.random.lognormal(0, self.sigma_int, size=(N, self.K))

        C = W
        A = intervention
        Y = (self.alpha * A + self.beta * C).sum(axis=1) + U
        return A, Y


@dataclass
class LinearSCMParams:
    K: int
    sigma_W: [list, np.ndarray]
    sigma_V: [list, np.ndarray]
    sigma_U: float
    alpha: [list, np.ndarray]
    beta: [list, np.ndarray]
    gamma: [list, np.ndarray]
    delta: [list, np.ndarray]  # Parameter for inter-action dependencies
    delta_mask: [list, np.ndarray]  # Mask for active edges
    sigma_int: [list, np.ndarray]
    p_edge: float  # Probability of an edge being active


def sample_linear_scm_params(K) -> LinearSCMParams:
    """
    Sample parameters for a Linear Structural Causal Model (SCM).

    Parameters
    ----------
    K : int
        Number of action variables.

    Returns
    -------
    LinearSCMParams
        Sampled parameters for the Linear SCM including coefficients,
        noise standard deviations, and edge probabilities.
    """
    sigma_W = np.random.uniform(0.5, 1.5, size=K)
    sigma_V = np.random.uniform(0.5, 1.5, size=K)
    sigma_U = np.random.uniform(0.5, 1.5)

    alpha = np.random.uniform(0.5, 1.5, size=K)
    beta = np.random.uniform(0.5, 1.5, size=K)
    gamma = np.random.uniform(0.5, 1.5, size=K)

    # Sample delta for inter-action dependencies
    delta = np.tril(np.random.uniform(-0.5, 0.5, size=(K, K)), k=-1)

    # Create a mask for active edges
    p_edge = 0.5  # Probability of an edge being active
    delta_mask = np.tril(np.random.binomial(1, p_edge, size=(K, K)), k=-1)

    sigma_int = np.random.uniform(0.5, 1.5)
    params = LinearSCMParams(
        K=K,
        sigma_W=sigma_W,
        sigma_V=sigma_V,
        sigma_U=sigma_U,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        delta_mask=delta_mask,
        sigma_int=sigma_int,
        p_edge=p_edge,
    )
    return params


def sample_linear_scm(K) -> LinearSCM:
    params = sample_linear_scm_params(K)
    return LinearSCM(**params.__dict__)
