"""Wasserstein Cluster: Enterprise-grade distribution clustering using Optimal Transport.

Provides Scikit-Learn compatible estimators and Wasserstein barycenters.
"""

from wasserstein_cluster.cluster.kmeans import WassersteinKMeans
from wasserstein_cluster.core.barycenter import compute_barycenter_1d
from wasserstein_cluster.core.distance import (
    pairwise_wasserstein_distance,
    wasserstein_distance_1d,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "WassersteinKMeans",
    "wasserstein_distance_1d",
    "pairwise_wasserstein_distance",
    "compute_barycenter_1d",
]
