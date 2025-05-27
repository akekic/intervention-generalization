from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class SCM(ABC):
    """
    Structural Causal Model (SCM) abstract class.

    There is one outcome variable Y and K action variables A_1, ..., A_K. Each action variable is a
    direct cause of Y (A_k -> Y). There are also K confounders C_1, ..., C_K, where each confounder
    is a direct cause of A_k (C_k -> A_k) and Y (C_k -> Y).

    The SCM is defined by the following structural equations:
        Y := sum_k f_k(A_k, C_k) + U
        A_k := g_k(C_k, A_1, ..., A_{k-1}, V_k)
        C_k := W_k
    where U, V_k, W_k are independent exogenous noise terms.
    """

    @abstractmethod
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
            and V has shape (N, K).
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
