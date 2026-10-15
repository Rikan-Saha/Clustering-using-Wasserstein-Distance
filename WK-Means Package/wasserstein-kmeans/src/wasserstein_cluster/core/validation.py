"""Input validation and normalization utilities for distribution data."""

from typing import Any, List, Tuple

import numpy as np


def validate_distribution_data(
    X: Any,
    allow_ragged: bool = True,
) -> Tuple[List[np.ndarray], bool]:
    """Validate and normalize input distribution data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or list of 1D array-like
        The input distributions.
    allow_ragged : bool, default=True
        Whether to allow distributions of differing lengths.

    Returns
    -------
    distributions : list of 1D np.ndarray of dtype float64
        Validated and converted distributions.
    is_ragged : bool
        True if distributions have differing sample sizes.

    Raises
    ------
    ValueError
        If input is empty, has invalid dimensions, or contains NaNs/Infs.
    TypeError
        If input cannot be converted to numerical arrays.
    """
    if X is None:
        raise ValueError("Input data X cannot be None.")

    # Handle pandas DataFrame
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            # Single distribution passed instead of collection of distributions
            raise ValueError(
                f"Expected 2D array or list of 1D distributions, got 1D array with shape {X.shape}. "
                "Reshape your data using [data] or data.reshape(1, -1) if it contains a single sample."
            )
        elif X.ndim == 2:
            if X.shape[0] == 0:
                raise ValueError("Input data X contains 0 distribution samples.")
            if X.shape[1] == 0:
                raise ValueError("Input distributions contain 0 sample points.")
            if not np.all(np.isfinite(X)):
                raise ValueError("Input data contains NaN, infinity, or non-finite values.")

            distributions = [np.asarray(row, dtype=np.float64) for row in X]
            return distributions, False
        else:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array with shape {X.shape}.")

    # Handle iterable / list of distributions
    try:
        raw_list = list(X)
    except TypeError as err:
        raise TypeError(
            f"Input X must be array-like or an iterable of distributions. Got {type(X).__name__}"
        ) from err

    if len(raw_list) == 0:
        raise ValueError("Input data X contains 0 distribution samples.")

    distributions = []
    lengths = set()

    for idx, item in enumerate(raw_list):
        arr = np.asarray(item, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                f"Distribution at index {idx} must be a 1D array of sample points, got shape {arr.shape}."
            )
        if arr.size == 0:
            raise ValueError(f"Distribution at index {idx} is empty (0 points).")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Distribution at index {idx} contains NaN or infinite values.")

        distributions.append(arr)
        lengths.add(arr.size)

    is_ragged = len(lengths) > 1

    if is_ragged and not allow_ragged:
        raise ValueError(
            f"Distributions have differing sample sizes: {lengths}. Ragged inputs are not permitted for this operation."
        )

    return distributions, is_ragged
