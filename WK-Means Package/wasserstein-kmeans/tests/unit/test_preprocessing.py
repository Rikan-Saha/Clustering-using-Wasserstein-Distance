"""Unit tests for distribution preprocessing."""

import numpy as np
import pytest

from wasserstein_cluster.preprocessing.distributions import (
    empirical_cdf,
    quantile_resample,
    resample_collection,
    standardize_distribution,
)


def test_empirical_cdf() -> None:
    samples = [3.0, 1.0, 2.0]
    vals, probs = empirical_cdf(samples)
    assert np.allclose(vals, [1.0, 2.0, 3.0])
    assert np.allclose(probs, [1.0 / 3, 2.0 / 3, 1.0])


def test_quantile_resample() -> None:
    samples = [0.0, 10.0]
    q = quantile_resample(samples, n_quantiles=5)
    assert len(q) == 5
    assert np.isclose(q[0], 0.0)
    assert np.isclose(q[-1], 10.0)
    assert np.isclose(q[2], 5.0)


def test_standardize_distribution() -> None:
    samples = np.array([10.0, 20.0, 30.0])
    std_samples = standardize_distribution(samples)
    assert np.isclose(np.mean(std_samples), 0.0)
    assert np.isclose(np.std(std_samples), 1.0)


def test_resample_collection() -> None:
    dists = [
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
    ]
    mat = resample_collection(dists, n_quantiles=10)
    assert mat.shape == (2, 10)
    assert np.all(np.diff(mat, axis=1) >= 0.0)


def test_invalid_samples_raise() -> None:
    with pytest.raises(ValueError, match="non-empty 1D array"):
        empirical_cdf([])
