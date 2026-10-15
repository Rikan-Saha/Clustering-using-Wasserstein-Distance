"""Interactive Bokeh visualizations for distribution clusters."""

from typing import Any, List

import numpy as np

from wasserstein_cluster.core.validation import validate_distribution_data


def plot_clusters_bokeh(
    data: Any,
    clusters: Any,
    bins: int = 20,
    width: int = 600,
    height: int = 350,
) -> Any:
    """Create interactive Bokeh histogram layout for clustered distributions.

    Parameters
    ----------
    data : array-like or list of 1D distributions
        Input distributions.
    clusters : 1D array-like
        Assigned cluster IDs.
    bins : int, default=20
        Number of histogram bins.
    width : int, default=600
        Figure width in pixels.
    height : int, default=350
        Figure height in pixels.

    Returns
    -------
    layout : bokeh.layouts.Column
        Interactive Bokeh column layout.
    """
    try:
        from bokeh.layouts import column
        from bokeh.plotting import figure
    except ImportError as err:
        raise ImportError(
            "Bokeh is required for interactive plotting. Install it via 'pip install wasserstein-cluster[viz]'."
        ) from err

    dists, _ = validate_distribution_data(data)
    lbls = np.asarray(clusters, dtype=np.int32)
    unique_clusters = sorted(set(lbls))

    plots: List[Any] = []

    for cluster_id in unique_clusters:
        p = figure(
            title=f"Cluster {cluster_id}",
            x_axis_label="Value",
            y_axis_label="Frequency",
            width=width,
            height=height,
        )

        cluster_samples = [dists[i] for i in range(len(dists)) if lbls[i] == cluster_id]
        for s_idx, dist in enumerate(cluster_samples):
            hist, edges = np.histogram(dist, bins=bins)
            p.quad(
                top=hist,
                bottom=0,
                left=edges[:-1],
                right=edges[1:],
                alpha=0.4,
                legend_label=f"Sample {s_idx}",
            )

        p.legend.click_policy = "hide"
        plots.append(p)

    return column(*plots)
