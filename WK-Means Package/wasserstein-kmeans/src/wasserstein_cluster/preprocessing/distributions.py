"""Distribution preprocessing, empirical CDF estimators, and quantile resampling."""

from typing import Any, Tuple

import numpy as np

from wasserstein_cluster.core.validation import validate_distribution_data


def empirical_cdf(samples: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical cumulative distribution function (ECDF) for 1D samples.

    Parameters
    ----------
    samples : 1D array-like
        Sample observations from the distribution.

    Returns
    -------
    sorted_values : np.ndarray
        Sorted values in ascending order.
    cumulative_probabilities : np.ndarray
        Empirical probabilities from 1/N to 1.0.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    sorted_vals = np.sort(arr)
    n = sorted_vals.size
    probabilities = np.linspace(1.0 / n, 1.0, n)
    return sorted_vals, probabilities


def quantile_resample(samples: Any, n_quantiles: int = 100) -> np.ndarray:
    """Resample an empirical distribution onto a uniform quantile grid.

    Parameters
    ----------
    samples : 1D array-like
        Sample observations from the distribution.
    n_quantiles : int, default=100
        Number of quantile points to evaluate across [0.0, 1.0].

    Returns
    -------
    quantiles : np.ndarray of shape (n_quantiles,)
        Evaluated quantile values sorted ascendingly.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")
    if n_quantiles <= 0:
        raise ValueError(f"n_quantiles must be >= 1, got {n_quantiles}.")

    grid = np.linspace(0.0, 1.0, n_quantiles)
    return np.asarray(np.quantile(arr, grid), dtype=np.float64)


def standardize_distribution(samples: Any, eps: float = 1e-8) -> np.ndarray:
    """Center distribution to zero mean and scale to unit variance.

    Parameters
    ----------
    samples : 1D array-like
        Sample observations.
    eps : float, default=1e-8
        Numerical epsilon to avoid division by zero for constant distributions.

    Returns
    -------
    standardized : np.ndarray of same shape
        Standardized sample values.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Samples must be a non-empty 1D array.")

    std = np.std(arr)
    if std < eps:
        return np.asarray(arr - np.mean(arr), dtype=np.float64)
    return np.asarray((arr - np.mean(arr)) / std, dtype=np.float64)


def resample_collection(distributions: Any, n_quantiles: int = 100) -> np.ndarray:
    """Resample a collection of (possibly ragged) distributions onto a uniform matrix.

    Parameters
    ----------
    distributions : array-like or list of 1D arrays
        Collection of distributions.
    n_quantiles : int, default=100
        Resolution of quantile grid for each distribution.

    Returns
    -------
    matrix : np.ndarray of shape (n_distributions, n_quantiles)
        Matrix of aligned quantile representations.
    """
    dists, _ = validate_distribution_data(distributions)
    n = len(dists)
    grid = np.linspace(0.0, 1.0, n_quantiles)

    matrix = np.empty((n, n_quantiles), dtype=np.float64)
    for i, d in enumerate(dists):
        matrix[i] = np.quantile(d, grid)

    return matrix
