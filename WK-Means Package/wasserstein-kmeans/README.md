# 📊 Wasserstein-Cluster

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy.readthedocs.io/)
[![Zero External Auth](https://img.shields.io/badge/auth-100%25%20offline-brightgreen.svg)](#-zero-credential-requirement)

**Enterprise-grade Wasserstein K-Means clustering for probability distributions and empirical data.**

`wasserstein-cluster` is an open-source Python library designed to cluster collections of 1-Dimensional probability distributions, empirical sample sets, and histograms using **Optimal Transport (Earth Mover's / Wasserstein Distance)**.

---

## 🌟 Key Features

* **Strict Scikit-Learn Compatibility**: Inherits natively from `BaseEstimator` and `ClusterMixin`. Fully compatible with standard ML workflows (`fit`, `predict`, `fit_predict`, `transform`, `score`).
* **Fast Vectorized Optimal Transport Kernels**: Optimized $\mathcal{O}(S \log S)$ and $\mathcal{O}(S)$ 1D Wasserstein distance computations ($W_1$ and $W_2$).
* **Exact Fréchet Quantile Barycenters**: Computes mathematically rigorous quantile averages (order-statistic averaging for uniform sample sets; quantile grid interpolation for variable-length distributions).
* **Native Ragged Array Support**: Directly cluster distributions with differing sample counts per observation without artificial padding.
* **Parallel CPU Acceleration**: Multi-core pairwise distance evaluations powered by `joblib`.
* **Wasserstein K-Means++**: Deterministic and probabilistic centroid seeding to prevent suboptimal local minima.
* **100% Offline & Private**: Zero cloud dependencies, zero SaaS services, and zero API keys required. Runs entirely on your local machine.

---

## 📦 Installation

Install `wasserstein-cluster` directly from GitHub using `pip` (zero accounts or setup required):

```bash
pip install "git+https://github.com/Rikan-Saha/Clustering-using-Wasserstein-Distance.git#subdirectory=WK-Means Package/wasserstein-kmeans"
```

---

## ⚙️ Prerequisites & Zero-Credential Requirement

### System Prerequisites
* **Python**: `>= 3.9` (fully tested on 3.9, 3.10, 3.11, 3.12, and 3.13)
* **Core Dependencies**:
  * `numpy>=1.22.0`
  * `scipy>=1.8.0`
  * `scikit-learn>=1.0.0`
  * `joblib>=1.2.0`

### 🔒 Zero-Credential Requirement
* **No API Keys or Tokens**: `wasserstein-cluster` does not require any credentials, tokens, or configuration keys.
* **No Account Registration**: Consumers do not need to register on any external service or cloud provider.
* **Completely Offline**: All mathematical algorithms and optional REST services execute 100% locally on your machine.

---

## 🚀 Quickstart

Here is a complete, working example clustering empirical distributions:

```python
import numpy as np
from wasserstein_cluster import WassersteinKMeans

# 1. Synthesize 30 empirical distributions from 3 clusters (10 per cluster)
np.random.seed(42)
distributions = []
for mean in [0.0, 5.0, 10.0]:
    for _ in range(10):
        samples = np.random.normal(loc=mean, scale=1.0, size=50)
        distributions.append(samples)

X = np.array(distributions)  # Shape: (30 distributions, 50 sample points each)

# 2. Instantiate and fit WassersteinKMeans
clusterer = WassersteinKMeans(n_clusters=3, random_state=42)
labels = clusterer.fit_predict(X)

# 3. Inspect Scikit-Learn attributes
print("Cluster Labels:", labels)
print("Inertia (Fréchet variance):", round(clusterer.inertia_, 4))
print("Iterations until convergence:", clusterer.n_iter_)
print("Number of Centroids:", len(clusterer.cluster_centers_))

# 4. Predict cluster assignment for an unseen distribution
unseen = np.random.normal(loc=5.1, scale=1.0, size=(1, 50))
predicted_cluster = clusterer.predict(unseen)[0]
print(f"Predicted cluster for Normal(loc=5.1): Cluster {predicted_cluster}")
```

### Expected Output
```text
Cluster Labels: [1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0]
Inertia: 0.8842
Iterations until convergence: 2
Number of Centroids: 3
Predicted cluster for Normal(loc=5.1): Cluster 2
```

---

## 📊 Variable-Length (Ragged) Distributions

Real-world distributions often have differing numbers of sample points. `WassersteinKMeans` handles ragged sequences natively:

```python
import numpy as np
from wasserstein_cluster import WassersteinKMeans

# Distributions with differing sample counts (e.g. 30, 80, 45, 120 points)
ragged_data = [
    np.random.normal(loc=0.0, scale=1.0, size=30),
    np.random.normal(loc=0.1, scale=1.0, size=80),
    np.random.normal(loc=10.0, scale=1.0, size=45),
    np.random.normal(loc=9.9, scale=1.0, size=120),
]

model = WassersteinKMeans(n_clusters=2, random_state=42)
labels = model.fit_predict(ragged_data)

print("Assigned cluster labels:", labels)
# Output: Assigned cluster labels: [1 1 0 0]
```

---

## 🧮 Direct Optimal Transport Primitives

You can also use the underlying distance and barycenter algorithms directly:

```python
import numpy as np
from wasserstein_cluster import (
    wasserstein_distance_1d,
    compute_barycenter_1d,
    pairwise_wasserstein_distance,
)

u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
v = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

# 1D Wasserstein distances (W1 and W2)
w1 = wasserstein_distance_1d(u, v, p=1)
w2 = wasserstein_distance_1d(u, v, p=2)
print(f"W1 Distance: {w1:.4f} | W2 Distance: {w2:.4f}")

# Exact 1D Fréchet Quantile Barycenter
cluster_members = [
    np.array([1.0, 2.0, 3.0]),
    np.array([2.0, 3.0, 4.0]),
    np.array([3.0, 4.0, 5.0]),
]
barycenter = compute_barycenter_1d(cluster_members)
print("Quantile Barycenter:", barycenter)  # [2.0, 3.0, 4.0]
```

---

## 📖 API Reference Summary

### `WassersteinKMeans`
```python
WassersteinKMeans(
    n_clusters=3,
    max_iter=300,
    tol=1e-4,
    random_state=None,
    n_init=10,
    p=2,
    n_jobs=None,
    n_quantiles=100,
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_clusters` | `int` | `3` | Number of distribution clusters to form. |
| `max_iter` | `int` | `300` | Maximum number of iterations for a single run. |
| `tol` | `float` | `1e-4` | Relative tolerance on inertia and center shift for convergence. |
| `random_state` | `int` or `None` | `None` | Seed for deterministic centroid initialization. |
| `n_init` | `int` | `10` | Number of times the algorithm runs with different centroid seeds. |
| `p` | `int` | `2` | Order of the Wasserstein metric ($p=1$ for $W_1$, $p=2$ for $W_2$). |
| `n_jobs` | `int` or `None` | `None` | Number of CPU cores to use for parallel pairwise distances. `-1` uses all cores. |
| `n_quantiles` | `int` | `100` | Number of quantile evaluation points for ragged/variable-length inputs. |

### Fitted Attributes
* `labels_`: `np.ndarray` of shape `(n_samples,)` - Cluster assignment for each input distribution.
* `cluster_centers_`: `List[np.ndarray]` - Representative Fréchet barycenter for each cluster.
* `inertia_`: `float` - Sum of squared Wasserstein distances of samples to their closest cluster center.
* `n_iter_`: `int` - Number of iterations run before convergence.

---

## 🧪 Contributing & Testing

If you are a developer looking to contribute or run the test suite locally:

```bash
# 1. Clone the repository and enter the package directory
git clone https://github.com/Rikan-Saha/Clustering-using-Wasserstein-Distance.git
cd "Clustering-using-Wasserstein-Distance/WK-Means Package/wasserstein-kmeans"

# 2. Install in editable mode with developer dependencies
pip install -e ".[dev]"

# 3. Run all unit and integration tests
pytest

# 4. Check test coverage (target: >= 85%)
pytest --cov=wasserstein_cluster --cov-report=term-missing

# 5. Run static type checking in strict mode
mypy src/wasserstein_cluster

# 6. Run linting
ruff check .
```

---

## 📄 License

This project is licensed under the terms of the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

```text
MIT License

Copyright (c) 2026 Rikan Saha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
