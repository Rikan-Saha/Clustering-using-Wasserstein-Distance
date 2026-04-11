from api.path_setup import *

from fastapi import FastAPI
from api.service import run_clustering
from api.schemas import ClusteringRequest, ClusteringResponse

app = FastAPI(title="Wasserstein KMeans API")


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/cluster", response_model=ClusteringResponse)
def cluster_data(request: ClusteringRequest):
    clusters = run_clustering(request.data, request.n_clusters)
    return {"clusters": clusters}