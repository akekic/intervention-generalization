import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def create_polynomial_model(
    n_order_fit,
    regularization="ridge",
    alpha_range=np.logspace(-8, -1, 8),
    max_iter=1000,
    tol=1e-4,
    n_jobs=None,
):
    if regularization.lower() == "ridge":
        regressor = Ridge(tol=tol)
    elif regularization.lower() == "lasso":
        regressor = Lasso(
            max_iter=max_iter,
            selection="random",
        )
    else:
        raise ValueError("regularization must be 'ridge' or 'lasso'")

    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=n_order_fit)),
            ("scaler", StandardScaler()),
            ("regularized_linear", regressor),
        ]
    )

    param_grid = {"regularized_linear__alpha": alpha_range}

    return GridSearchCV(model, param_grid, cv=3, n_jobs=n_jobs)


def get_polynomial_coefficients(model) -> tuple[dict, float]:
    """
    Extract polynomial coefficients and intercept from a fitted polynomial model.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        A fitted pipeline containing PolynomialFeatures and a linear regressor.

    Returns
    -------
    Tuple[dict, float]
        A tuple containing:
        - dict: Dictionary mapping feature names to their coefficients
        - float: The intercept term
    """
    poly_features = model.named_steps["poly"]
    linear_regression = model.named_steps["regularized_linear"]
    feature_names = poly_features.get_feature_names_out()
    coefficients = linear_regression.coef_
    intercept = linear_regression.intercept_

    return dict(zip(feature_names, coefficients)), intercept
