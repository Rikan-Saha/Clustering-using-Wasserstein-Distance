"""Unit tests for 1D Wasserstein Barycenter (Fréchet mean)."""

import numpy as np
import pytest

from wasserstein_cluster.core.barycenter import compute_barycenter_1d


def test_barycenter_of_identical_distributions() -> None:
    dist = np.array([1.0, 2.0, 3.0, 4.0])
    bary = compute_barycenter_1d([dist, dist, dist])
    assert np.allclose(bary, dist)


def test_barycenter_of_two_shifted_distributions() -> None:
    # Midpoint between [0, 2] and [4, 6] should be [2, 4]
    d1 = np.array([0.0, 2.0])
    d2 = np.array([4.0, 6.0])
    bary = compute_barycenter_1d([d1, d2])
    assert np.allclose(bary, np.array([2.0, 4.0]))


def test_barycenter_weighted() -> None:
    d1 = np.array([0.0, 0.0])
    d2 = np.array([10.0, 10.0])
    # 80% weight on d1, 20% on d2 -> 0.8*0 + 0.2*10 = 2.0
    bary = compute_barycenter_1d([d1, d2], weights=[0.8, 0.2])
    assert np.allclose(bary, np.array([2.0, 2.0]))


def test_barycenter_ragged_distributions() -> None:
    d1 = np.array([1.0, 1.0, 1.0, 1.0])
    d2 = np.array([3.0, 3.0])
    bary = compute_barycenter_1d([d1, d2], n_quantiles=10)
    assert len(bary) == 10
    # Average of constant 1.0 and constant 3.0 is constant 2.0
    assert np.allclose(bary, 2.0)


def test_barycenter_invalid_weights() -> None:
    d1 = np.array([1.0, 2.0])
    d2 = np.array([3.0, 4.0])
    with pytest.raises(ValueError, match="Weights must be 1D array of length 2"):
        compute_barycenter_1d([d1, d2], weights=[1.0, 2.0, 3.0])
