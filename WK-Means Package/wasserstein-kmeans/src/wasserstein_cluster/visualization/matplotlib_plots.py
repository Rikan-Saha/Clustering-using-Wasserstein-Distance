"""Publication-quality Matplotlib plots for distribution clusters and barycenters."""

from typing import Any, Optional, Tuple

import numpy as np

from wasserstein_cluster.core.validation import validate_distribution_data


def plot_clusters_distribution(
    X: Any,
    labels: Any,
    cluster_centers: Optional[Any] = None,
    figsize: Tuple[int, int] = (10, 6),
    alpha: float = 0.4,
) -> Any:
    """Plot distribution samples grouped by cluster with overlayed barycenters.

    Parameters
    ----------
    X : array-like or list of 1D distributions
        Distribution samples.
    labels : 1D array-like
        Cluster labels for each distribution sample.
    cluster_centers : list of 1D arrays, optional
        Fitted cluster barycenter centroids to highlight.
    figsize : tuple of (int, int), default=(10, 6)
        Matplotlib figure dimensions.
    alpha : float, default=0.4
        Transparency for sample distributions.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Matplotlib figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "Matplotlib is required for static plotting. Install it via 'pip install wasserstein-cluster[viz]'."
        ) from err

    dists, _ = validate_distribution_data(X)
    lbls = np.asarray(labels, dtype=np.int32)
    unique_clusters = sorted(set(lbls))

    fig, axes = plt.subplots(
        nrows=len(unique_clusters),
        ncols=1,
        figsize=(figsize[0], figsize[1] * len(unique_clusters) // 2),
        squeeze=False,
    )

    for idx, c_id in enumerate(unique_clusters):
        ax = axes[idx, 0]
        c_samples = [dists[i] for i in range(len(dists)) if lbls[i] == c_id]

        for s in c_samples:
            sorted_s = np.sort(s)
            y_coords = np.linspace(0, 1, len(sorted_s))
            ax.plot(sorted_s, y_coords, color="tab:blue", alpha=alpha, linewidth=1.0)

        if cluster_centers is not None and c_id < len(cluster_centers):
            centroid = np.sort(cluster_centers[c_id])
            c_y = np.linspace(0, 1, len(centroid))
            ax.plot(
                centroid,
                c_y,
                color="tab:red",
                linewidth=2.5,
                label=f"Cluster {c_id} Barycenter",
            )

        ax.set_title(f"Cluster {c_id} (N={len(c_samples)} distributions)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Empirical CDF")
        ax.grid(True, linestyle="--", alpha=0.5)
        if cluster_centers is not None:
            ax.legend(loc="upper left")

    plt.tight_layout()
    return fig
