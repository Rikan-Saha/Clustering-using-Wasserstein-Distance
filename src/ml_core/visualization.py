from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import Select, CustomJS
import numpy as np


def plot_clusters_bokeh(data, clusters):
    """
    Interactive cluster visualization using Bokeh
    """

    unique_clusters = sorted(set(clusters))

    plots = {}

    for cluster_id in unique_clusters:

        p = figure(
            title=f"Cluster {cluster_id}",
            x_axis_label="Value",
            y_axis_label="Frequency",
            width=600,
            height=400
        )

        for i, dist in enumerate(data):
            if clusters[i] == cluster_id:
                hist, edges = np.histogram(dist, bins=20)

                p.quad(
                    top=hist,
                    bottom=0,
                    left=edges[:-1],
                    right=edges[1:],
                    alpha=0.5,
                    legend_label=f"Sample {i}"
                )

        plots[str(cluster_id)] = p

    # Show only first cluster initially
    layout = column(*plots.values())

    show(layout)