"""FastAPI application for Wasserstein K-Means inference."""

from fastapi import FastAPI, HTTPException, status

from api.schemas import ClusteringRequest, ClusteringResponse
from api.service import run_clustering

app = FastAPI(
    title="Wasserstein KMeans API",
    description="Enterprise REST service for Optimal Transport clustering of probability distributions.",
    version="0.1.0",
)


@app.get("/")
def home():
    """Health check and API metadata endpoint."""
    return {
        "message": "API is running",
        "version": "0.1.0",
        "engine": "wasserstein-cluster",
    }


@app.post(
    "/cluster",
    response_model=ClusteringResponse,
    status_code=status.HTTP_200_OK,
    summary="Cluster empirical distributions",
)
def cluster_data(request: ClusteringRequest):
    """Cluster a collection of 1D distributions using Wasserstein distance.

    Returns the assigned cluster labels, inertia, and iteration count.
    """
    try:
        result = run_clustering(
            data=request.data,
            n_clusters=request.n_clusters,
            max_iter=request.max_iter,
            random_state=request.random_state,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clustering execution failure: {str(exc)}",
        ) from exc
