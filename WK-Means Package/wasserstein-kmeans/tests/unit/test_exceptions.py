"""Unit tests for custom warnings and exceptions."""

import numpy as np
import pytest

from wasserstein_cluster import WassersteinKMeans
from wasserstein_cluster.exceptions import ConvergenceWarning


def test_convergence_warning_emitted() -> None:
    # Set max_iter=1 so algorithm stops before convergence, triggering ConvergenceWarning
    rng = np.random.RandomState(42)
    data = [rng.normal(loc=float(i % 5), scale=1.0, size=20) for i in range(25)]

    model = WassersteinKMeans(n_clusters=5, max_iter=1, tol=1e-12, random_state=42)
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        model.fit(data)
