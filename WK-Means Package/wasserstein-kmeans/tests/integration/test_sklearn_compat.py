"""Compatibility tests verifying compliance with Scikit-Learn estimator conventions."""

from sklearn.base import clone

from wasserstein_cluster import WassersteinKMeans


def test_sklearn_estimator_clone() -> None:
    """Verify that WassersteinKMeans can be cloned without side-effects."""
    model = WassersteinKMeans(n_clusters=4, max_iter=150, tol=1e-5, random_state=123)
    cloned_model = clone(model)

    assert cloned_model.n_clusters == 4
    assert cloned_model.max_iter == 150
    assert cloned_model.tol == 1e-5
    assert cloned_model.random_state == 123
    assert not hasattr(cloned_model, "cluster_centers_")


def test_sklearn_get_and_set_params() -> None:
    """Verify Scikit-Learn parameter getters and setters."""
    model = WassersteinKMeans(n_clusters=2)
    params = model.get_params()
    assert params["n_clusters"] == 2
    assert params["max_iter"] == 300

    model.set_params(n_clusters=5, max_iter=500)
    assert model.n_clusters == 5
    assert model.max_iter == 500


def test_estimator_representation() -> None:
    """Verify clean __repr__ string representation from BaseEstimator."""
    model = WassersteinKMeans(n_clusters=4, max_iter=200, random_state=42)
    repr_str = repr(model)
    assert "WassersteinKMeans" in repr_str
    assert "n_clusters=4" in repr_str
    assert "max_iter=200" in repr_str
