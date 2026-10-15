"""Core mathematical algorithms: distances, barycenters, and initializations."""

from wasserstein_cluster.core.barycenter import compute_barycenter_1d
from wasserstein_cluster.core.distance import (
    pairwise_wasserstein_distance,
    wasserstein_distance_1d,
)
from wasserstein_cluster.core.initialization import init_centroids
from wasserstein_cluster.core.validation import validate_distribution_data

__all__ = [
    "wasserstein_distance_1d",
    "pairwise_wasserstein_distance",
    "compute_barycenter_1d",
    "init_centroids",
    "validate_distribution_data",
]
