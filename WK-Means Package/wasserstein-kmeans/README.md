# 📊 Wasserstein-Cluster

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy.readthedocs.io/)

**Enterprise-grade Wasserstein K-Means clustering for probability distributions and empirical data.**

Full Scikit-Learn Estimator API compliance (`BaseEstimator`, `ClusterMixin`), fast vectorized 1D Optimal Transport distance kernels, Fréchet quantile barycenters, and multi-core `joblib` parallel acceleration.

---

## 🚀 Quickstart

```bash
pip install wasserstein-cluster
```

```python
import numpy as np
from wasserstein_cluster import WassersteinKMeans

# Generate sample 1D distributions (e.g. 100 distributions, 50 points each)
data = [np.random.normal(loc=i % 3, scale=1.0, size=50) for i in range(100)]

# Initialize and fit
clusterer = WassersteinKMeans(n_clusters=3, random_state=42)
clusterer.fit(data)

# Inspect standard Scikit-Learn properties
print("Labels:", clusterer.labels_)
print("Centroids (Barycenters):", clusterer.cluster_centers_)
print("Inertia:", clusterer.inertia_)
```

---

## 📦 Optional Extras

* **Interactive & publication visualizations**:
  ```bash
  pip install wasserstein-cluster[viz]
  ```
* **Production FastAPI deployment microservice**:
  ```bash
  pip install wasserstein-cluster[api]
  ```
* **Full development suite**:
  ```bash
  pip install wasserstein-cluster[all]
  ```

---

## 📄 License
MIT License.
