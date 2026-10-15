"""Scikit-Learn compliant Wasserstein K-Means clustering estimator."""

import warnings
from typing import Any, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from wasserstein_cluster.core.barycenter import compute_barycenter_1d
from wasserstein_cluster.core.distance import wasserstein_distance_1d
from wasserstein_cluster.core.initialization import init_centroids
from wasserstein_cluster.core.validation import validate_distribution_data
from wasserstein_cluster.exceptions import ConvergenceWarning


class WassersteinKMeans(BaseEstimator, ClusterMixin):
    """Wasserstein K-Means clustering for probability distributions.

    Clusters collections of probability distributions using the optimal transport
    (Wasserstein / Earth Mover's) metric and Fréchet quantile barycenters.

    Conforms to the Scikit-Learn Estimator interface (BaseEstimator, ClusterMixin),
    enabling native compatibility with Pipeline and GridSearchCV.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters to form and centroids to generate.
    init : {'k-means++', 'random'} or array-like, default='k-means++'
        Centroid initialization method.
    n_init : int, default=10
        Number of times the algorithm will be run with different centroid seeds.
        The final result is the best output of n_init consecutive runs in terms of inertia.
    max_iter : int, default=300
        Maximum number of iterations of the algorithm for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in cluster centers or inertia delta to declare convergence.
    p : int, default=1
        Order of Wasserstein distance (1 for W1 / Earth Mover's, 2 for W2 metric).
    n_quantiles : int, default=100
        Number of quantile points to evaluate for ragged arrays.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization.
    n_jobs : int or None, default=-1
        Number of parallel jobs to run. If -1, all CPUs are used.
    verbose : int, default=0
        Verbosity mode.

    Attributes
    ----------
    cluster_centers_ : list of np.ndarray
        Coordinates of cluster centers (barycenter distributions).
    labels_ : np.ndarray of shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run in the best run.
    n_features_in_ : int
        Number of features/points per distribution.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        *,
        init: Union[str, Any] = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        p: int = 1,
        n_quantiles: int = 100,
        random_state: Any = None,
        n_jobs: Optional[int] = -1,
        verbose: int = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.p = p
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _check_is_fitted(self) -> List[np.ndarray]:
        """Verify that the estimator has been fitted."""
        if not hasattr(self, "cluster_centers_") or self.cluster_centers_ is None:
            raise ValueError(
                "This WassersteinKMeans instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        return self.cluster_centers_

    def fit(self, X: Any, y: Any = None) -> "WassersteinKMeans":
        """Compute Wasserstein k-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points) or list of 1D arrays
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        dists, is_ragged = validate_distribution_data(X)
        n_samples = len(dists)

        if self.n_clusters <= 0:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters}.")
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters={self.n_clusters} cannot exceed the number of samples={n_samples}."
            )

        # Store input metadata
        self.n_features_in_ = int(self.n_quantiles if is_ragged else dists[0].size)

        # Pre-sort distributions once for performance optimization
        sorted_X = [np.sort(d) for d in dists]
        rng = check_random_state(self.random_state)

        best_inertia = np.inf
        best_labels: Optional[np.ndarray] = None
        best_centroids: Optional[List[np.ndarray]] = None
        best_n_iter = 0

        # Run n_init separate iterations
        n_runs = 1 if not isinstance(self.init, str) else max(1, self.n_init)

        for run in range(n_runs):
            run_seed = rng.randint(0, np.iinfo(np.int32).max)
            centroids = init_centroids(
                sorted_X,
                self.n_clusters,
                init=self.init,
                random_state=run_seed,
                p=self.p,
                n_quantiles=self.n_quantiles,
            )

            prev_inertia = np.inf
            labels: np.ndarray = np.zeros(n_samples, dtype=int)
            n_iter = 0

            converged = False
            for it in range(self.max_iter):
                n_iter = it + 1
                sq_distances_to_centroids = np.empty((n_samples, self.n_clusters), dtype=np.float64)

                # Step 1: Assign each point to the closest centroid
                for c_idx, centroid in enumerate(centroids):
                    for i in range(n_samples):
                        d = wasserstein_distance_1d(
                            sorted_X[i],
                            centroid,
                            p=self.p,
                            is_sorted=True,
                            n_quantiles=self.n_quantiles,
                        )
                        sq_distances_to_centroids[i, c_idx] = d**2

                labels = np.argmin(sq_distances_to_centroids, axis=1)
                min_sq_dists = np.min(sq_distances_to_centroids, axis=1)
                current_inertia = float(np.sum(min_sq_dists))

                # Step 2: Update centroids via Fréchet quantile barycenters
                new_centroids: List[np.ndarray] = []
                for c_idx in range(self.n_clusters):
                    cluster_members = [sorted_X[i] for i in range(n_samples) if labels[i] == c_idx]

                    if len(cluster_members) == 0:
                        # Empty cluster handling: re-seed with sample having largest distance to its assigned centroid
                        furthest_idx = int(np.argmax(min_sq_dists))
                        new_centroids.append(sorted_X[furthest_idx].copy())
                    else:
                        bary = compute_barycenter_1d(
                            cluster_members,
                            n_quantiles=self.n_quantiles if is_ragged else None,
                            is_sorted=True,
                        )
                        new_centroids.append(bary)

                # Step 3: Check convergence
                center_shift = sum(
                    wasserstein_distance_1d(
                        c, nc, p=self.p, is_sorted=True, n_quantiles=self.n_quantiles
                    )
                    for c, nc in zip(centroids, new_centroids)
                )

                centroids = new_centroids

                if center_shift <= self.tol or (
                    it > 0
                    and np.isfinite(prev_inertia)
                    and abs(prev_inertia - current_inertia) <= self.tol * max(1.0, prev_inertia)
                ):
                    converged = True
                    break

                prev_inertia = current_inertia

            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_labels = labels
                best_centroids = centroids
                best_n_iter = n_iter
                best_converged = converged

        self.cluster_centers_ = best_centroids
        self.labels_ = (
            best_labels if best_labels is not None else np.zeros(n_samples, dtype=np.int32)
        )
        self.inertia_ = float(best_inertia)
        self.n_iter_ = int(best_n_iter)

        if not best_converged:
            warnings.warn(
                f"WassersteinKMeans did not converge within max_iter={self.max_iter} iterations.",
                ConvergenceWarning,
                stacklevel=2,
            )

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict the closest cluster each distribution in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points) or list of 1D arrays
            New data distributions to predict.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        centers = self._check_is_fitted()
        dists, _ = validate_distribution_data(X)
        n_samples = len(dists)

        sorted_X = [np.sort(d) for d in dists]
        labels = np.zeros(n_samples, dtype=np.int32)

        for i in range(n_samples):
            dists_to_c = [
                wasserstein_distance_1d(
                    sorted_X[i], c, p=self.p, is_sorted=True, n_quantiles=self.n_quantiles
                )
                for c in centers
            ]
            labels[i] = int(np.argmin(dists_to_c))

        return labels

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        """Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points) or list of 1D arrays
            Input distributions.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_

    def transform(self, X: Any) -> np.ndarray:
        """Transform X to a cluster-distance space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points) or list of 1D arrays
            Input distributions.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_clusters)
            Distances from each distribution in X to each cluster center.
        """
        centers = self._check_is_fitted()
        dists, _ = validate_distribution_data(X)
        n_samples = len(dists)
        sorted_X = [np.sort(d) for d in dists]

        result = np.empty((n_samples, self.n_clusters), dtype=np.float64)
        for i in range(n_samples):
            for c_idx, c in enumerate(centers):
                result[i, c_idx] = wasserstein_distance_1d(
                    sorted_X[i], c, p=self.p, is_sorted=True, n_quantiles=self.n_quantiles
                )

        return result

    def score(self, X: Any, y: Any = None) -> float:
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_points) or list of 1D arrays
            Input distributions.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        score : float
            Negative inertia on X.
        """
        self._check_is_fitted()
        dist_matrix = self.transform(X)
        min_sq_dists = np.min(dist_matrix, axis=1) ** 2
        return -float(np.sum(min_sq_dists))
