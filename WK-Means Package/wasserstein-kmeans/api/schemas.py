"""Pydantic schemas for the clustering API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ClusteringRequest(BaseModel):
    data: List[List[float]] = Field(
        ...,
        description="List of empirical probability distribution samples.",
        min_length=1,
    )
    n_clusters: int = Field(
        default=3,
        ge=1,
        description="Number of clusters to generate.",
    )
    max_iter: int = Field(
        default=300,
        ge=1,
        description="Maximum iterations for the clustering loop.",
    )
    random_state: Optional[int] = Field(
        default=None,
        description="Random seed for deterministic initialization.",
    )


class ClusteringResponse(BaseModel):
    clusters: List[int] = Field(
        ..., description="Assigned cluster index for each distribution sample."
    )
    inertia: Optional[float] = Field(
        None, description="Sum of squared Wasserstein distances to centroids."
    )
    n_iter: Optional[int] = Field(None, description="Number of iterations run.")
