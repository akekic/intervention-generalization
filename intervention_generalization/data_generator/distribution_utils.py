from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats


class DistributionType(Enum):
    GAUSSIAN = "gaussian"
    # LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    # GAMMA = "gamma"
    LOGISTIC = "logistic"


@dataclass
class DistributionParams:
    dist_type: DistributionType
    mean: float
    std: float


def get_distribution(dist_params: DistributionParams) -> stats.rv_continuous:
    """
    Create a scipy.stats distribution object from distribution parameters.

    Parameters
    ----------
    dist_params : DistributionParams
        Parameters specifying the distribution type, mean, and standard deviation.

    Returns
    -------
    scipy.stats distribution
        A scipy.stats distribution object configured with the specified parameters.

    Raises
    ------
    ValueError
        If the distribution type is not supported.
    """
    if dist_params.dist_type == DistributionType.GAUSSIAN:
        return stats.norm(loc=dist_params.mean, scale=dist_params.std)
    # elif dist_params.dist_type == DistributionType.LOGNORMAL:
    #     mu = np.log(
    #         dist_params.mean**2
    #         / np.sqrt(dist_params.std**2 + dist_params.mean**2)
    #     )
    #     sigma = np.sqrt(np.log(1 + (dist_params.std / dist_params.mean) ** 2))
    #     return stats.lognorm(s=sigma, scale=np.exp(mu))
    elif dist_params.dist_type == DistributionType.UNIFORM:
        a = dist_params.mean - np.sqrt(3) * dist_params.std
        b = dist_params.mean + np.sqrt(3) * dist_params.std
        return stats.uniform(loc=a, scale=b - a)
    # elif dist_params.dist_type == DistributionType.GAMMA:
    #     shape = (dist_params.mean / dist_params.std) ** 2
    #     scale = dist_params.std**2 / dist_params.mean
    #     return stats.gamma(a=shape, scale=scale)
    elif dist_params.dist_type == DistributionType.LOGISTIC:
        # For logistic distribution:
        # mean = location
        # std = scale * pi / sqrt(3)
        location = dist_params.mean
        scale = dist_params.std * np.sqrt(3) / np.pi
        return stats.logistic(loc=location, scale=scale)
    else:
        raise ValueError(f"Unknown distribution type: {dist_params.dist_type}")


def sample_from_distribution(dist_params: DistributionParams, size: int):
    dist = get_distribution(dist_params)
    return dist.rvs(size=size)
