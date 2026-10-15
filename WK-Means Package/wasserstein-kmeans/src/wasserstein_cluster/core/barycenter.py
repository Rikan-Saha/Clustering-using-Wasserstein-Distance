"""Wasserstein Barycenter (Fréchet Mean) algorithms for 1D distributions."""

from typing import Any, Optional

import numpy as np

from wasserstein_cluster.core.validation import validate_distribution_data


def compute_barycenter_1d(
    distributions: Any,
    weights: Optional[Any] = None,
    n_quantiles: Optional[int] = None,
    is_sorted: bool = False,
) -> np.ndarray:
    """Compute the exact Wasserstein Barycenter (Fréchet mean) for 1D distributions.

    In 1-Dimension under the Wasserstein metric, the barycenter of a collection of
    probability distributions equals the weighted average of their inverse CDFs
    (quantile functions):

        F_{barycenter}^{-1}(t) = sum_j lambda_j F_j^{-1}(t)

    For empirical samples of equal size, this is evaluated exactly by averaging
    their sorted order statistics. For ragged (varying-length) distributions,
    quantiles are sampled across a uniform grid.

    Parameters
    ----------
    distributions : array-like of shape (n_distributions, n_points) or list of 1D arrays
        Collection of distributions belonging to the cluster.
    weights : 1D array-like of shape (n_distributions,), optional
        Importance weights for each distribution. If None, uniform weighting (1/K) is used.
    n_quantiles : int, optional
        Resolution for quantile evaluation. If None and distributions have equal length,
        preserves the original sample size.
    is_sorted : bool, default=False
        Whether input distributions are already sorted in ascending order.

    Returns
    -------
    barycenter : np.ndarray of shape (n_points,) or (n_quantiles,)
        The resulting Wasserstein barycenter empirical distribution, sorted ascendingly.
    """
    dists, is_ragged = validate_distribution_data(distributions)
    K = len(dists)

    # Normalized weights
    if weights is None:
        w = np.full(K, 1.0 / K, dtype=np.float64)
    else:
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.ndim != 1 or w_arr.size != K:
            raise ValueError(f"Weights must be 1D array of length {K}, got shape {w_arr.shape}.")
        if np.any(w_arr < 0):
            raise ValueError("Weights must be non-negative.")
        w_sum = np.sum(w_arr)
        if w_sum <= 0:
            raise ValueError("Sum of weights must be strictly positive.")
        w = w_arr / w_sum

    if K == 1:
        single = dists[0]
        if n_quantiles is not None and n_quantiles != single.size:
            grid = np.linspace(0.0, 1.0, n_quantiles)
            return np.asarray(np.quantile(single, grid), dtype=np.float64)
        return single if is_sorted else np.sort(single)

    # Uniform-length fast path (exact order-statistic averaging)
    if not is_ragged and (n_quantiles is None or n_quantiles == dists[0].size):
        sorted_matrix = np.empty((K, dists[0].size), dtype=np.float64)
        for i, d in enumerate(dists):
            sorted_matrix[i] = d if is_sorted else np.sort(d)

        # Weighted dot-product across rows
        barycenter = np.dot(w, sorted_matrix)
        return np.asarray(barycenter, dtype=np.float64)

    # Ragged distributions or custom quantile resolution
    q_len = n_quantiles if n_quantiles is not None else 100
    grid = np.linspace(0.0, 1.0, q_len)

    quantiles_matrix = np.empty((K, q_len), dtype=np.float64)
    for i, d in enumerate(dists):
        quantiles_matrix[i] = np.quantile(d, grid)

    barycenter = np.dot(w, quantiles_matrix)
    return np.asarray(barycenter, dtype=np.float64)
