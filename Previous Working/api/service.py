import sys

from src.ml_core.kmeans_wasserstein import WassersteinKMeans


def run_clustering(data, n_clusters):
    model = WassersteinKMeans(n_clusters=n_clusters)
    clusters = model.fit(data)
    return clusters.tolist()