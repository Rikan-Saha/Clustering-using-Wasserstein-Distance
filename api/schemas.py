from pydantic import BaseModel
from typing import List


class ClusteringRequest(BaseModel):
    data: List[List[float]]   # list of distributions
    n_clusters: int = 3

class ClusteringResponse(BaseModel):
    clusters: List[int]