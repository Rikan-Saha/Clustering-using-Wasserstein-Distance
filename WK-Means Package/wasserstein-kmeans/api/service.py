"""Service layer for Wasserstein clustering API."""

from typing import Any, Dict, List, Optional

from wasserstein_cluster import WassersteinKMeans


def run_clustering(
    data: List[List[float]],
    n_clusters: int = 3,
    max_iter: int = 300,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Wasserstein K-Means clustering on distribution data.

    Parameters
    ----------
    data : list of list of float
        The empirical distributions.
    n_clusters : int, default=3
        Number of clusters.
    max_iter : int, default=300
        Maximum iterations.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing cluster labels, inertia, and iteration count.
    """
    model = WassersteinKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = model.fit_predict(data)
    return {
        "clusters": labels.tolist(),
        "inertia": model.inertia_,
        "n_iter": model.n_iter_,
    }
