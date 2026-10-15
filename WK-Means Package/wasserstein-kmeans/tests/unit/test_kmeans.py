"""Unit tests for WassersteinKMeans estimator."""

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from wasserstein_cluster import WassersteinKMeans


def test_kmeans_basic_clustering() -> None:
    # 2 clearly separated distribution clusters
    # Cluster 0: Centered around 0.0
    # Cluster 1: Centered around 10.0
    rng = np.random.RandomState(42)
    cluster_0 = [rng.normal(loc=0.0, scale=0.5, size=30) for _ in range(5)]
    cluster_1 = [rng.normal(loc=10.0, scale=0.5, size=30) for _ in range(5)]
    data = cluster_0 + cluster_1

    model = WassersteinKMeans(n_clusters=2, random_state=42)
    model.fit(data)

    assert hasattr(model, "cluster_centers_")
    assert len(model.cluster_centers_) == 2
    assert model.labels_.shape == (10,)
    assert model.inertia_ >= 0.0
    assert model.n_iter_ >= 1
    assert model.n_features_in_ == 30

    # Points 0..4 should have same label, 5..9 should have opposite label
    assert len(set(model.labels_[:5])) == 1
    assert len(set(model.labels_[5:])) == 1
    assert model.labels_[0] != model.labels_[5]


def test_kmeans_predict_and_transform() -> None:
    rng = np.random.RandomState(42)
    train_data = [rng.normal(loc=0.0, scale=0.2, size=20) for _ in range(4)] + [
        rng.normal(loc=5.0, scale=0.2, size=20) for _ in range(4)
    ]

    model = WassersteinKMeans(n_clusters=2, random_state=42)
    model.fit(train_data)

    # Test prediction on new unseen distributions
    test_d0 = [rng.normal(loc=0.1, scale=0.2, size=20)]
    test_d1 = [rng.normal(loc=4.9, scale=0.2, size=20)]

    pred0 = model.predict(test_d0)
    pred1 = model.predict(test_d1)

    assert pred0[0] == model.labels_[0]
    assert pred1[0] == model.labels_[4]

    # Test transform
    dist_matrix = model.transform(test_d0)
    assert dist_matrix.shape == (1, 2)
    assert np.isclose(dist_matrix[0, pred0[0]], 0.0, atol=0.5)

    # Test score
    score_val = model.score(test_d0)
    assert score_val <= 0.0


def test_kmeans_pipeline_and_gridsearch() -> None:
    rng = np.random.RandomState(0)
    X = [rng.normal(loc=float(i % 2 * 5), scale=0.5, size=15) for i in range(8)]

    # Test Pipeline
    pipe = Pipeline([("clusterer", WassersteinKMeans(n_clusters=2, random_state=0))])
    pipe.fit(X)
    labels = pipe.named_steps["clusterer"].labels_
    assert len(labels) == 8

    # Test GridSearchCV
    grid = GridSearchCV(
        WassersteinKMeans(random_state=0),
        param_grid={"n_clusters": [2, 3]},
        cv=2,
    )
    # Using dummy y for scikit-learn CV splitting
    dummy_y = [0, 0, 0, 0, 1, 1, 1, 1]
    grid.fit(X, dummy_y)
    assert grid.best_params_["n_clusters"] in [2, 3]


def test_kmeans_not_fitted_error() -> None:
    model = WassersteinKMeans()
    with pytest.raises(ValueError, match="not fitted yet"):
        model.predict([[1.0, 2.0]])
