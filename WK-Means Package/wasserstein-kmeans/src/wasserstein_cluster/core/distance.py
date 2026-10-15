"""Fast 1D Wasserstein distance algorithms and pairwise distance matrices."""

from typing import Any, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance as scipy_wasserstein_1d

from wasserstein_cluster.core.validation import validate_distribution_data


def wasserstein_distance_1d(
    u: Any,
    v: Any,
    p: int = 1,
    is_sorted: bool = False,
    n_quantiles: int = 100,
) -> float:
    """Compute the 1-Dimensional p-Wasserstein distance between two distributions.

    For 1D empirical distributions of equal size, the Wasserstein distance
    equals the Lp norm of their sorted order statistics. For distributions of
    differing sizes, continuous quantile interpolation across [0, 1] is evaluated.

    Parameters
    ----------
    u : 1D array-like
        First empirical distribution sample.
    v : 1D array-like
        Second empirical distribution sample.
    p : int, default=1
        Order of Wasserstein distance (1 for Earth Mover's distance, 2 for W2 metric).
    is_sorted : bool, default=False
        If True, assumes u and v are already sorted in ascending order,
        skipping sorting overhead for O(S) runtime.
    n_quantiles : int, default=100
        Number of quantile points to evaluate if distributions have differing lengths.

    Returns
    -------
    distance : float
        The p-Wasserstein distance between u and v.
    """
    if p not in (1, 2):
        raise ValueError(f"Only p=1 and p=2 Wasserstein distances are supported, got p={p}.")

    arr_u = np.asarray(u, dtype=np.float64)
    arr_v = np.asarray(v, dtype=np.float64)

    if arr_u.ndim != 1 or arr_v.ndim != 1:
        raise ValueError(
            f"Inputs must be 1D distributions. Got u shape {arr_u.shape}, v shape {arr_v.shape}."
        )
    if arr_u.size == 0 or arr_v.size == 0:
        raise ValueError("Inputs must not be empty.")

    # Optimized fast-path: equal length distributions
    if arr_u.size == arr_v.size:
        u_sorted = arr_u if is_sorted else np.sort(arr_u)
        v_sorted = arr_v if is_sorted else np.sort(arr_v)

        diff = np.abs(u_sorted - v_sorted)
        if p == 1:
            return float(np.mean(diff))
        else:  # p == 2
            return float(np.sqrt(np.mean(diff**2)))

    # Unequal length distributions:
    if p == 1:
        # SciPy provides an exact, analytically exact 1D W1 implementation
        return float(scipy_wasserstein_1d(arr_u, arr_v))
    else:  # p == 2
        # Quantile interpolation across uniform grid
        grid = np.linspace(0.0, 1.0, n_quantiles)
        q_u = np.quantile(arr_u, grid)
        q_v = np.quantile(arr_v, grid)
        return float(np.sqrt(np.mean((q_u - q_v) ** 2)))


def pairwise_wasserstein_distance(
    X: Any,
    Y: Optional[Any] = None,
    p: int = 1,
    n_jobs: Optional[int] = None,
    is_sorted: bool = False,
    n_quantiles: int = 100,
) -> np.ndarray:
    """Compute the pairwise p-Wasserstein distance matrix between collections of distributions.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features) or list of 1D distributions
        First collection of distributions.
    Y : array-like of shape (n_samples_Y, n_features) or list of 1D distributions, optional
        Second collection of distributions. If None, computes pairwise distances within X.
    p : int, default=1
        Order of Wasserstein distance (1 or 2).
    n_jobs : int, optional
        Number of CPU workers for parallel execution via joblib. If None or 1, runs single-threaded.
    is_sorted : bool, default=False
        If True, assumes distributions are already sorted in ascending order.
    n_quantiles : int, default=100
        Number of quantile points to evaluate for ragged arrays.

    Returns
    -------
    dist_matrix : np.ndarray of shape (n_samples_X, n_samples_Y)
        Pairwise distance matrix where dist_matrix[i, j] = W_p(X[i], Y[j]).
    """
    dists_X, _ = validate_distribution_data(X)
    symmetric = Y is None

    if symmetric:
        dists_Y = dists_X
    else:
        dists_Y, _ = validate_distribution_data(Y)

    nX = len(dists_X)
    nY = len(dists_Y)

    # Sort distributions once up-front if not already sorted, optimizing O(N*S log S) to one-time O(N * S log S)
    if not is_sorted:
        sorted_X = [np.sort(x) for x in dists_X]
        sorted_Y = sorted_X if symmetric else [np.sort(y) for y in dists_Y]
    else:
        sorted_X = dists_X
        sorted_Y = dists_Y

    matrix = np.zeros((nX, nY), dtype=np.float64)

    # Sequential execution for small sample sets or n_jobs == 1
    if n_jobs == 1 or (nX * nY < 500 and (n_jobs is None or n_jobs == 1)):
        if symmetric:
            for i in range(nX):
                for j in range(i + 1, nX):
                    d = wasserstein_distance_1d(
                        sorted_X[i], sorted_Y[j], p=p, is_sorted=True, n_quantiles=n_quantiles
                    )
                    matrix[i, j] = d
                    matrix[j, i] = d
        else:
            for i in range(nX):
                for j in range(nY):
                    matrix[i, j] = wasserstein_distance_1d(
                        sorted_X[i], sorted_Y[j], p=p, is_sorted=True, n_quantiles=n_quantiles
                    )
        return matrix

    # Parallel execution via joblib
    def _compute_row(i: int) -> Tuple[int, np.ndarray]:
        row = np.zeros(nY, dtype=np.float64)
        start_j = (i + 1) if symmetric else 0
        for j in range(start_j, nY):
            row[j] = wasserstein_distance_1d(
                sorted_X[i], sorted_Y[j], p=p, is_sorted=True, n_quantiles=n_quantiles
            )
        return i, row

    effective_jobs = -1 if n_jobs is None else n_jobs
    results = Parallel(n_jobs=effective_jobs, prefer="threads")(
        delayed(_compute_row)(i) for i in range(nX)
    )

    for i, row in results:
        matrix[i, :] = row

    if symmetric:
        # Symmetrize
        matrix = matrix + matrix.T

    return matrix
