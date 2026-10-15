"""Unit tests for input validation utilities."""

import numpy as np
import pytest

from wasserstein_cluster.core.validation import validate_distribution_data


def test_validate_valid_2d_array() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    dists, is_ragged = validate_distribution_data(X)
    assert len(dists) == 2
    assert not is_ragged
    assert np.allclose(dists[0], [1.0, 2.0])


def test_validate_valid_ragged_list() -> None:
    X = [[1.0, 2.0, 3.0], [4.0, 5.0]]
    dists, is_ragged = validate_distribution_data(X)
    assert len(dists) == 2
    assert is_ragged


def test_validate_ragged_disallowed_raises() -> None:
    X = [[1.0, 2.0, 3.0], [4.0, 5.0]]
    with pytest.raises(ValueError, match="Ragged inputs are not permitted"):
        validate_distribution_data(X, allow_ragged=False)


def test_validate_nan_raises() -> None:
    X = [[1.0, np.nan], [3.0, 4.0]]
    with pytest.raises(ValueError, match="contains NaN"):
        validate_distribution_data(X)


def test_validate_empty_raises() -> None:
    with pytest.raises(ValueError, match="contains 0 distribution samples"):
        validate_distribution_data([])
