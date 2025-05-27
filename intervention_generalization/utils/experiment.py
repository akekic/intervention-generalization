from typing import Union, Iterable

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from ..data_generator import SCM
from ..model import create_polynomial_model


def run_experiment(
    scm: SCM,
    K: int,
    N: Union[int, Iterable[int]],
    n_order_fit: int,
    regularization: str,
    skip_obs: bool = False,
    skip_method: bool = False,
    skip_jint: bool = False,
    skip_pool: bool = False,
    verbose: bool = False,
    n_jobs: int = None,
) -> dict[str, float]:
    """
    Run a complete intervention generalization experiment.

    This function generates data from the given SCM, trains various models,
    and evaluates their performance on joint intervention prediction.

    Parameters
    ----------
    scm : SCM
        The structural causal model to use for data generation.
    K : int
        Number of action variables.
    N : Union[int, Iterable[int]]
        Number of samples. If int, uses the same number for all data types.
        If iterable of 3 integers, uses [N_obs, N_sint, N_jint].
    n_order_fit : int
        Order of polynomial features to use in model fitting.
    regularization : str
        Type of regularization ('ridge' or 'lasso').
    skip_obs : bool, optional
        Whether to skip the observational-only model, by default False.
    skip_method : bool, optional
        Whether to skip the intervention generalization method, by default False.
    skip_jint : bool, optional
        Whether to skip the joint intervention baseline, by default False.
    skip_pool : bool, optional
        Whether to skip the pooled data baseline, by default False.
    verbose : bool, optional
        Whether to print detailed model information, by default False.
    n_jobs : int, optional
        Number of parallel jobs for model fitting, by default None.

    Returns
    -------
    dict
        Dictionary containing RMSE results for each method:
        - 'rmse_obs': RMSE of observational-only model (if not skipped)
        - 'rmse_method': RMSE of intervention generalization method (if not skipped)
        - 'rmse_jint': RMSE of joint intervention baseline (if not skipped)
        - 'rmse_pool': RMSE of pooled data baseline (if not skipped)

    Raises
    ------
    ValueError
        If N is not an integer or an iterable of 3 integers.
    """
    if isinstance(N, int):
        N_obs = N_sint = N_jint = N
    elif isinstance(N, Iterable) and len(list(N)) == 3:
        N_obs, N_sint, N_jint = N
    else:
        raise ValueError("N must be either an integer or an iterable of 3 integers")

    result = {}

    # Generate and process observational data
    if not (skip_obs and skip_method and skip_pool):
        A_obs, Y_obs = scm.generate_obs(N_obs)
        A_obs_train, A_obs_test, Y_obs_train, Y_obs_test = train_test_split(
            A_obs, Y_obs, test_size=0.2, random_state=42
        )
        model_obs = create_polynomial_model(
            n_order_fit, regularization=regularization, n_jobs=n_jobs
        )
        model_obs.fit(A_obs_train, Y_obs_train)
        if verbose:
            print(f"Obs model best params: {model_obs.best_params_}")
        del A_obs, Y_obs, A_obs_train, Y_obs_train  # Free up memory

    # Process single intervention data
    if not (skip_method and skip_pool):
        models_sint = []
        for k in range(K):
            A, Y = scm.generate_single_int(k, N_sint)
            A_train, _, Y_train, _ = train_test_split(
                A, Y, test_size=0.2, random_state=42
            )
            model_sint = create_polynomial_model(
                n_order_fit, regularization=regularization, n_jobs=n_jobs
            )
            model_sint.fit(A_train, Y_train)
            models_sint.append(model_sint)
            if verbose:
                print(f"Sint model {k} best params: {model_sint.best_params_}")
            del A, Y, A_train, Y_train  # Free up memory

    # Generate and process joint intervention data
    A_jint, Y_jint = scm.generate_joint_int(N_jint)
    _, A_jint_test, _, Y_jint_test = train_test_split(
        A_jint, Y_jint, test_size=0.2, random_state=42
    )

    # Compute predictions and metrics
    if not (skip_obs and skip_method and skip_pool):
        Y_jint_pred_obs = model_obs.predict(A_jint_test)

    if not skip_method:
        Y_jint_pred_method = np.zeros_like(Y_jint_test)
        for model_sint in models_sint:
            Y_jint_pred_method += model_sint.predict(A_jint_test)
        Y_jint_pred_method -= (K - 1) * Y_jint_pred_obs
        mse_method = mean_squared_error(Y_jint_pred_method, Y_jint_test)
        result["rmse_method"] = np.sqrt(mse_method)

    if not skip_obs:
        mse_obs = mean_squared_error(Y_jint_pred_obs, Y_jint_test)
        result["rmse_obs"] = np.sqrt(mse_obs)

    if not skip_jint:
        A_jint_train, _, Y_jint_train, _ = train_test_split(
            A_jint, Y_jint, test_size=0.2, random_state=42
        )
        model_jint = create_polynomial_model(
            n_order_fit, regularization=regularization, n_jobs=n_jobs
        )
        model_jint.fit(A_jint_train, Y_jint_train)
        if verbose:
            print(f"Jint model best params: {model_jint.best_params_}")
        Y_jint_pred = model_jint.predict(A_jint_test)
        mse_jint = mean_squared_error(Y_jint_pred, Y_jint_test)
        result["rmse_jint"] = np.sqrt(mse_jint)
        del A_jint_train, Y_jint_train  # Free up memory

    if not skip_pool:
        Y_pool_pred = np.zeros_like(Y_jint_test)
        chunk_size = min(N_obs, N_sint, 10000)  # Process data in chunks

        # Process observational data
        for i in range(0, N_obs, chunk_size):
            A_obs_chunk, Y_obs_chunk = scm.generate_obs(min(chunk_size, N_obs - i))
            model_pool = create_polynomial_model(
                n_order_fit, regularization=regularization, n_jobs=n_jobs
            )
            model_pool.fit(A_obs_chunk, Y_obs_chunk)
            Y_pool_pred += model_pool.predict(A_jint_test)
            del A_obs_chunk, Y_obs_chunk  # Free up memory

        # Process single intervention data
        for k in range(K):
            for i in range(0, N_sint, chunk_size):
                A_sint_chunk, Y_sint_chunk = scm.generate_single_int(
                    k, min(chunk_size, N_sint - i)
                )
                model_pool.fit(A_sint_chunk, Y_sint_chunk)
                Y_pool_pred += model_pool.predict(A_jint_test)
                del A_sint_chunk, Y_sint_chunk  # Free up memory

        Y_pool_pred /= 1 + K  # Average the predictions
        mse_pool = mean_squared_error(Y_pool_pred, Y_jint_test)
        result["rmse_pool"] = np.sqrt(mse_pool)

    del A_jint, Y_jint, A_jint_test, Y_jint_test  # Free up remaining memory

    return result
