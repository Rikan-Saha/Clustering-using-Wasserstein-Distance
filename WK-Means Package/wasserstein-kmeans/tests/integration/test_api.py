"""End-to-end integration tests for the FastAPI inference service."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client() -> TestClient:
    """Fixture providing FastAPI TestClient."""
    return TestClient(app)


def test_api_health_check(client: TestClient) -> None:
    """Validate GET / root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "API is running"
    assert data["version"] == "0.1.0"
    assert data["engine"] == "wasserstein-cluster"


def test_api_cluster_valid_request(client: TestClient) -> None:
    """Validate POST /cluster with clearly separated distribution samples."""
    # 2 clusters: 3 samples centered around 0.0, 3 samples centered around 10.0
    rng = np.random.RandomState(42)
    c0 = [rng.normal(loc=0.0, scale=0.2, size=25).tolist() for _ in range(3)]
    c1 = [rng.normal(loc=10.0, scale=0.2, size=25).tolist() for _ in range(3)]
    payload = {
        "data": c0 + c1,
        "n_clusters": 2,
        "max_iter": 100,
        "random_state": 42,
    }

    response = client.post("/cluster", json=payload)
    assert response.status_code == 200
    res = response.json()

    assert "clusters" in res
    assert "inertia" in res
    assert "n_iter" in res
    assert len(res["clusters"]) == 6
    assert res["inertia"] >= 0.0
    assert res["n_iter"] >= 1

    # Samples 0, 1, 2 must share same cluster label
    # Samples 3, 4, 5 must share same cluster label
    labels = res["clusters"]
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_api_cluster_synthetic_csv_data(client: TestClient) -> None:
    """Validate POST /cluster using synthetic_data.csv from the project."""
    # synthetic_data.csv is at Project6/data/synthetic_data.csv
    csv_path = (
        Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "synthetic_data.csv"
    )
    if not csv_path.exists():
        pytest.skip("synthetic_data.csv not found at expected path")

    import pandas as pd

    df = pd.read_csv(csv_path)
    data = df.values.tolist()

    payload = {
        "data": data,
        "n_clusters": 2,
        "random_state": 42,
    }

    response = client.post("/cluster", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert len(res["clusters"]) == len(data)


def test_api_error_handling_clusters_exceed_samples(client: TestClient) -> None:
    """Validate 400 Bad Request when n_clusters > n_samples."""
    payload = {
        "data": [[1.0, 2.0], [3.0, 4.0]],
        "n_clusters": 5,
    }
    response = client.post("/cluster", json=payload)
    assert response.status_code == 400
    assert "cannot exceed the number of samples" in response.json()["detail"]


def test_api_error_handling_invalid_n_clusters(client: TestClient) -> None:
    """Validate 422 Unprocessable Entity when n_clusters < 1."""
    payload = {
        "data": [[1.0, 2.0], [3.0, 4.0]],
        "n_clusters": 0,
    }
    response = client.post("/cluster", json=payload)
    assert response.status_code == 422


def test_api_error_handling_empty_data(client: TestClient) -> None:
    """Validate 422 Unprocessable Entity when data is empty."""
    payload = {
        "data": [],
        "n_clusters": 2,
    }
    response = client.post("/cluster", json=payload)
    assert response.status_code == 422


def test_api_error_handling_nan_in_distribution(client: TestClient) -> None:
    """Validate 400 Bad Request when distribution contains non-finite values."""
    payload = {
        "data": [[1.0, 2.0], [3.0, None]],  # JSON null translates to None / NaN
        "n_clusters": 2,
    }
    response = client.post("/cluster", json=payload)
    # Pydantic or validation logic will catch non-float or NaN
    assert response.status_code in (400, 422)


def test_api_404_not_found(client: TestClient) -> None:
    """Validate 404 for unknown endpoints."""
    response = client.get("/non_existent_endpoint")
    assert response.status_code == 404
