"""Unit tests for centroid initialization."""

import numpy as np
import pytest

from wasserstein_cluster.core.initialization import init_centroids


def test_kmeans_plus_plus_reproducible() -> None:
    X = [
        np.array([1.0, 1.2]),
        np.array([1.1, 0.9]),
        np.array([10.0, 10.2]),
        np.array([9.9, 10.1]),
    ]
    c1 = init_centroids(X, n_clusters=2, init="k-means++", random_state=42)
    c2 = init_centroids(X, n_clusters=2, init="k-means++", random_state=42)
    assert len(c1) == 2
    assert np.allclose(c1[0], c2[0])
    assert np.allclose(c1[1], c2[1])


def test_random_initialization() -> None:
    X = [np.array([float(i)]) for i in range(10)]
    centroids = init_centroids(X, n_clusters=3, init="random", random_state=0)
    assert len(centroids) == 3


def test_invalid_n_clusters() -> None:
    X = [np.array([1.0, 2.0])]
    with pytest.raises(ValueError, match="n_clusters=3 cannot exceed number of samples=1"):
        init_centroids(X, n_clusters=3)
