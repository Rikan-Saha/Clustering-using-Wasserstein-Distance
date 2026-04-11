import numpy as np
from ml_core.distance import compute_wasserstein_distance

class WassersteinKMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def _initialize_centroids(self, data):
        indices = np.random.choice(len(data), self.n_clusters, replace=False)
        return [data[i] for i in indices]

    def _assign_clusters(self, data):
        clusters = []
        for point in data:
            distances = [
                compute_wasserstein_distance(point, centroid)
                for centroid in self.centroids
            ]
            clusters.append(np.argmin(distances))
        return np.array(clusters)

    def _update_centroids(self, data, clusters):
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_points = [data[j] for j in range(len(data)) if clusters[j] == i]

            if len(cluster_points) == 0:
                new_centroids.append(self.centroids[i])
            else:
                new_centroids.append(np.mean(cluster_points, axis=0))

        return new_centroids

    def fit(self, data):

        self.centroids = self._initialize_centroids(data)

        for _ in range(self.max_iter):
            clusters = self._assign_clusters(data)
            new_centroids = self._update_centroids(data, clusters)

            if all(np.array_equal(c, nc) for c, nc in zip(self.centroids, new_centroids)):
                break

            self.centroids = new_centroids

        return clusters