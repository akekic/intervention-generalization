import warnings
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Tuple, Callable, List

import numpy as np

from .distribution_utils import (
    DistributionParams,
    sample_from_distribution,
    DistributionType,
)
from .scm import SCM


def random_polynomial_factory(
    n_order: int, n_dim: int, scale: float = 0.1, normalize: bool = True
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a random multivariate polynomial function, supporting vectorized input.

    Parameters
    ----------
    n_order : int
        The order of the polynomial.
    n_dim : int
        The number of dimensions (variables) in the polynomial.
    scale : float, optional
        The scaling factor for the polynomial coefficients, by default 0.1.
    normalize : bool, optional
        Whether to normalize the coefficients to sum to 1, by default True.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A function that computes the polynomial given an input array.

    Notes
    -----
    The polynomial is defined as:

    .. math::
        P(x) = \sum_{i=1}^{M} c_i \prod_{j=1}^{n} x_j^{p_{ij}}

    where \(M\) is the number of terms, \(c_i\) are the coefficients, \(x_j\) are the input variables,
    and \(p_{ij}\) are the powers.

    Scaling is implemented as:

    .. math::
        c_i = \text{scale} \cdot \text{randn}()

    where \text{randn} is the standard normal distribution.

    If normalization is enabled, the coefficients are normalized as:

    .. math::
        c_i = \frac{c_i}{\sum_{i=1}^{M} |c_i|}
    """
    all_powers = list(product(range(n_order + 1), repeat=n_dim))
    powers = [p for p in all_powers if sum(p) <= n_order]
    coeffs = scale * np.random.randn(len(powers))

    if normalize:
        # Normalize coefficients to sum to 1
        coeffs = coeffs / np.sum(np.abs(coeffs))

    def polynomial(x: np.ndarray) -> np.ndarray:
        """
        Compute the polynomial for the given input.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (..., n_dim).

        Returns
        -------
        np.ndarray
            Output array of shape (...).
        """
        if x.shape[-1] != n_dim:
            raise ValueError(
                f"Last dimension of input must be {n_dim}, got {x.shape[-1]}"
            )

        result = np.zeros(x.shape[:-1])
        for coeff, power_combo in zip(coeffs, powers):
            term = np.full(x.shape[:-1], coeff)
            for dim, power in enumerate(power_combo):
                term *= x[..., dim] ** power
            result += term

        return result

    polynomial.coeffs = coeffs
    polynomial.powers = powers
    return polynomial


class PolynomialSCM(SCM):
    """
    Polynomial Structural Causal Model (SCM) class.

    Parameters
    ----------
    K : int
        Number of action variables.
    W_params : Iterable[DistributionParams]
        Parameters for the exogenous noise terms W of the confounders.
    V_params : Iterable[DistributionParams]
        Parameters for the exogenous noise terms V of the action variables.
    U_params : DistributionParams
        Parameters for the U exogenous noise term of the outcome variable.
    delta_mask : Iterable[Iterable[float]]
        Mask for the delta values.
    sigma_int : float
        Standard deviation for the intervention noise.
    p_edge : float
        Probability of an edge in the delta mask.
    n_order : int, optional
        Order of the polynomial, by default 2.
    scale : float, optional
        Scaling factor for the polynomial coefficients, by default 0.1.
    """

    def __init__(
        self,
        K: int,
        W_params: Iterable[DistributionParams],
        V_params: Iterable[DistributionParams],
        U_params: DistributionParams,
        delta_mask: Iterable[Iterable[float]],
        sigma_int: float,
        p_edge: float,
        n_order: int = 2,
        scale: float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.W_params = W_params
        self.V_params = V_params
        self.U_params = U_params
        self.delta_mask = np.array(delta_mask)
        self.sigma_int = sigma_int
        self.p_edge = p_edge

        self.f = [
            random_polynomial_factory(n_order, 2, scale, normalize=True)
            for _ in range(K)
        ]
        self.g = [
            random_polynomial_factory(
                n_order, 2 + np.sum(self.delta_mask[k, :k]), scale, normalize=True
            )
            for k in range(self.K)
        ]

        # Store observational statistics for each action variable
        self.action_means = np.zeros(K)
        self.action_stds = np.zeros(K)
        self.obs_generated = False

    def generate_noise(self, N: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate independent exogenous noise terms.

        Parameters
        ----------
        N : int, optional
            Number of samples to generate, by default 100.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Generated noise terms (W, U, V) where W has shape (N, K), U has shape (N,),
            and V has shape (N, K). Distributions are determined by the respective
            DistributionParams (W_params, U_params, V_params).
        """
        W = np.array(
            [sample_from_distribution(params, N) for params in self.W_params]
        ).T
        V = np.array(
            [sample_from_distribution(params, N) for params in self.V_params]
        ).T
        U = sample_from_distribution(self.U_params, N)
        return W, U, V

    def generate_obs(self, N: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate observational data.

        Parameters
        ----------
        N : int, optional
            Number of samples to generate, by default 100.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Generated observational data (A, Y).
        """
        W, U, V = self.generate_noise(N)

        C = W
        A = np.zeros((N, self.K))
        for k in range(self.K):
            parents = self.delta_mask[k, :] == 1
            g_k_input = np.concatenate([A[:, parents], C[:, [k]], V[:, [k]]], axis=-1)
            A[:, k] = self.g[k](g_k_input)

        # Store observational statistics
        self.action_means = A.mean(axis=0)
        self.action_stds = A.std(axis=0)
        self.obs_generated = True

        Y = (
            np.sum(
                [self.f[k](np.stack([A[:, k], C[:, k]]).T) for k in range(self.K)],
                axis=0,
            )
            + U
        )
        return A, Y

    def generate_single_int(
        self, intervention_var: int = 0, N: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for a single intervention.

        Parameters
        ----------
        intervention_var : int, optional
            Index of the intervention variable, by default 0.
        N : int, optional
            Number of samples to generate, by default 100.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Generated data for the single intervention (A, Y).
        """
        if not self.obs_generated:
            warnings.warn(
                "Observational data has not been generated yet. Run generate_obs()"
                " first for more accurate interventional distributions.",
                UserWarning,
            )
            # Use default values if obs data hasn't been generated
            intervention = np.random.normal(0, self.sigma_int, N)
        else:
            intervention = np.random.normal(
                self.action_means[intervention_var],
                self.action_stds[intervention_var],
                N,
            )

        W, U, V = self.generate_noise(N)

        C = W
        A = np.zeros((N, self.K))
        for k in range(self.K):
            if k == intervention_var:
                A[:, k] = intervention
            else:
                parents = self.delta_mask[k, :] == 1
                g_k_input = np.concatenate(
                    [A[:, parents], C[:, [k]], V[:, [k]]], axis=-1
                )
                A[:, k] = self.g[k](g_k_input)

        Y = (
            np.sum(
                [self.f[k](np.stack([A[:, k], C[:, k]]).T) for k in range(self.K)],
                axis=0,
            )
            + U
        )
        return A, Y

    def generate_joint_int(self, N: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for joint interventions.

        Parameters
        ----------
        N : int, optional
            Number of samples to generate, by default 100.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Generated data for joint interventions (A, Y).
        """
        if not self.obs_generated:
            warnings.warn(
                "Observational data has not been generated yet. Run generate_obs()"
                " first for more accurate interventional distributions.",
                UserWarning,
            )
            # Use default values if obs data hasn't been generated
            intervention = np.random.normal(0, self.sigma_int, size=(N, self.K))
        else:
            intervention = np.random.normal(
                self.action_means, self.action_stds, size=(N, self.K)
            )

        W, U, V = self.generate_noise(N)

        C = W
        A = intervention
        Y = (
            np.sum(
                [self.f[k](np.stack([A[:, k], C[:, k]]).T) for k in range(self.K)],
                axis=0,
            )
            + U
        )
        return A, Y


@dataclass
class PolynomialSCMParams:
    """
    Parameters for the Polynomial Structural Causal Model (SCM).

    Attributes
    ----------
    K : int
        Number of action variables.
    W_params : List[DistributionParams]
        Parameters for the W distributions.
    V_params : List[DistributionParams]
        Parameters for the V distributions.
    U_params : DistributionParams
        Parameters for the U distribution.
    delta_mask : Iterable[Iterable[float]]
        Mask for the delta values.
    sigma_int : float
        Standard deviation for the intervention noise.
    p_edge : float
        Probability of an edge in the delta mask.
    n_order : int, optional
        Order of the polynomial, by default 2.
    scale : float, optional
        Scaling factor for the polynomial coefficients, by default 0.1.
    """

    K: int
    W_params: List[DistributionParams]
    V_params: List[DistributionParams]
    U_params: DistributionParams
    delta_mask: Iterable[Iterable[float]]
    sigma_int: float
    p_edge: float
    n_order: int = 2
    scale: float = 0.1


def sample_polynomial_scm_params(K) -> PolynomialSCMParams:
    """
    Sample parameters for a Polynomial Structural Causal Model (SCM).

    Parameters
    ----------
    K : int
        Number of action variables.

    Returns
    -------
    PolynomialSCMParams
        Sampled parameters for the Polynomial SCM.
    """
    W_params = []
    V_params = []
    for _ in range(K):
        W_params.append(
            DistributionParams(
                dist_type=np.random.choice(list(DistributionType)),
                mean=0,
                std=np.random.uniform(0.5, 0.5),
            )
        )
        V_params.append(
            DistributionParams(
                dist_type=np.random.choice(list(DistributionType)),
                mean=0,
                std=np.random.uniform(0.1, 0.1),
            )
        )

    U_params = DistributionParams(
        dist_type=np.random.choice(list(DistributionType)),
        mean=0,
        std=np.random.uniform(0.1, 0.1),
    )

    p_edge = 0.3
    delta_mask = np.tril(np.random.binomial(1, p_edge, size=(K, K)), k=-1)
    sigma_int = np.random.uniform(0.1, 0.1)
    scale = 0.1
    n_order = 2

    params = PolynomialSCMParams(
        K=K,
        W_params=W_params,
        V_params=V_params,
        U_params=U_params,
        delta_mask=delta_mask,
        sigma_int=sigma_int,
        p_edge=p_edge,
        n_order=n_order,
        scale=scale,
    )
    return params


def sample_polynomial_scm(K) -> PolynomialSCM:
    """
    Sample a Polynomial Structural Causal Model (SCM).

    Parameters
    ----------
    K : int
        Number of action variables.

    Returns
    -------
    PolynomialSCM
        Sampled Polynomial SCM.
    """
    params = sample_polynomial_scm_params(K)
    return PolynomialSCM(**params.__dict__)
