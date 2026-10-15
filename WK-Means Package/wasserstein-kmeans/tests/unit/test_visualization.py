"""Unit tests for visualization routines."""

import numpy as np

from wasserstein_cluster.visualization.bokeh_plots import plot_clusters_bokeh
from wasserstein_cluster.visualization.matplotlib_plots import plot_clusters_distribution


def test_matplotlib_plot_creation() -> None:
    data = [np.array([1.0, 2.0]), np.array([5.0, 6.0])]
    labels = [0, 1]
    centers = [np.array([1.5, 1.5]), np.array([5.5, 5.5])]

    fig = plot_clusters_distribution(data, labels, cluster_centers=centers)
    assert fig is not None


def test_bokeh_plot_creation() -> None:
    data = [np.array([1.0, 2.0, 3.0]), np.array([5.0, 6.0, 7.0])]
    labels = [0, 1]

    layout = plot_clusters_bokeh(data, labels)
    assert layout is not None
