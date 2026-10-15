"""Unit tests for 1D Wasserstein distances and pairwise distance computation."""

import numpy as np
import pytest

from wasserstein_cluster.core.distance import (
    pairwise_wasserstein_distance,
    wasserstein_distance_1d,
)


def test_wasserstein_distance_identical_distributions() -> None:
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dist = wasserstein_distance_1d(u, u, p=1)
    assert np.isclose(dist, 0.0)

    dist_w2 = wasserstein_distance_1d(u, u, p=2)
    assert np.isclose(dist_w2, 0.0)


def test_wasserstein_distance_shifted_dirac() -> None:
    # W1 between two shifted distributions by constant c equals |c|
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([3.0, 4.0, 5.0])  # Shifted by +2.0
    w1 = wasserstein_distance_1d(u, v, p=1)
    assert np.isclose(w1, 2.0)

    w2 = wasserstein_distance_1d(u, v, p=2)
    assert np.isclose(w2, 2.0)


def test_wasserstein_distance_ragged_sizes() -> None:
    u = np.array([0.0, 0.0, 0.0, 0.0])  # Point mass at 0
    v = np.array([2.0, 2.0])  # Point mass at 2
    w1 = wasserstein_distance_1d(u, v, p=1)
    assert np.isclose(w1, 2.0)

    w2 = wasserstein_distance_1d(u, v, p=2)
    assert np.isclose(w2, 2.0)


def test_pairwise_wasserstein_distance_symmetric() -> None:
    X = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
    ]
    mat = pairwise_wasserstein_distance(X, p=1)
    assert mat.shape == (3, 3)
    assert np.allclose(mat, mat.T)
    assert np.allclose(np.diag(mat), 0.0)
    assert np.isclose(mat[0, 1], 3.0)
    assert np.isclose(mat[0, 2], 6.0)


def test_pairwise_wasserstein_distance_asymmetric() -> None:
    X = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    Y = [np.array([1.0, 2.0]), np.array([5.0, 6.0]), np.array([9.0, 10.0])]
    mat = pairwise_wasserstein_distance(X, Y, p=1)
    assert mat.shape == (2, 3)
    assert np.isclose(mat[0, 0], 0.0)
    assert np.isclose(mat[0, 1], 4.0)
    assert np.isclose(mat[1, 1], 2.0)


def test_invalid_parameters_raise_error() -> None:
    with pytest.raises(ValueError, match="Only p=1 and p=2"):
        wasserstein_distance_1d([1, 2], [3, 4], p=3)
