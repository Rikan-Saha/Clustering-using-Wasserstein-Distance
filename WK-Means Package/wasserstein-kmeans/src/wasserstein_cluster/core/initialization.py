"""Centroid initialization algorithms, including Wasserstein K-Means++."""

from typing import Any, List, Union

import numpy as np
from sklearn.utils import check_random_state

from wasserstein_cluster.core.distance import wasserstein_distance_1d
from wasserstein_cluster.core.validation import validate_distribution_data


def init_centroids(
    X: Any,
    n_clusters: int,
    init: Union[str, Any] = "k-means++",
    random_state: Any = None,
    p: int = 1,
    n_quantiles: int = 100,
) -> List[np.ndarray]:
    """Initialize cluster centroids using random or Wasserstein K-Means++ selection.

    Parameters
    ----------
    X : array-like or list of 1D distributions
        Input dataset distributions.
    n_clusters : int
        Number of centroids to initialize.
    init : {'k-means++', 'random'} or array-like, default='k-means++'
        Method of initialization or pre-specified centroids.
    random_state : int, RandomState instance or None, default=None
        Random number generator seed.
    p : int, default=1
        Order of Wasserstein distance for distance weighting.
    n_quantiles : int, default=100
        Quantile resolution for distance comparisons on ragged arrays.

    Returns
    -------
    centroids : list of 1D np.ndarray
        List of initialized centroid distributions (sorted).
    """
    dists, _ = validate_distribution_data(X)
    n_samples = len(dists)

    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}.")
    if n_clusters > n_samples:
        raise ValueError(f"n_clusters={n_clusters} cannot exceed number of samples={n_samples}.")

    # Pre-sort copies for consistent distance calculations
    sorted_dists = [np.sort(d) for d in dists]
    rng = check_random_state(random_state)

    # Pre-computed initial centroids passed directly
    if not isinstance(init, str):
        custom_centroids, _ = validate_distribution_data(init)
        if len(custom_centroids) != n_clusters:
            raise ValueError(
                f"Custom init provided {len(custom_centroids)} centroids, expected {n_clusters}."
            )
        return [np.sort(c) for c in custom_centroids]

    if init == "random":
        chosen_indices = rng.choice(n_samples, size=n_clusters, replace=False)
        return [sorted_dists[i].copy() for i in chosen_indices]

    if init != "k-means++":
        raise ValueError(
            f"Unknown init strategy '{init}'. Expected 'k-means++', 'random', or array-like."
        )

    # Wasserstein K-Means++ initialization
    centroids: List[np.ndarray] = []

    # Step 1: Pick first centroid uniformly at random
    first_idx = int(rng.randint(0, n_samples))
    centroids.append(sorted_dists[first_idx].copy())

    # Distances from each sample to closest existing centroid
    closest_sq_dists = np.full(n_samples, np.inf, dtype=np.float64)

    # Step 2: Iteratively pick remaining centroids
    for _ in range(1, n_clusters):
        latest_centroid = centroids[-1]
        for i in range(n_samples):
            d = wasserstein_distance_1d(
                sorted_dists[i], latest_centroid, p=p, is_sorted=True, n_quantiles=n_quantiles
            )
            d_sq = d**2
            if d_sq < closest_sq_dists[i]:
                closest_sq_dists[i] = d_sq

        total_sq_dist = np.sum(closest_sq_dists)
        if total_sq_dist <= 0:
            # Degenerate case: all remaining points are identical to existing centroids
            remaining_indices = [i for i in range(n_samples) if sorted_dists[i] not in centroids]
            if len(remaining_indices) > 0:
                next_idx = int(rng.choice(remaining_indices))
            else:
                next_idx = int(rng.randint(0, n_samples))
        else:
            probs = closest_sq_dists / total_sq_dist
            next_idx = int(rng.choice(n_samples, p=probs))

        centroids.append(sorted_dists[next_idx].copy())

    return centroids
