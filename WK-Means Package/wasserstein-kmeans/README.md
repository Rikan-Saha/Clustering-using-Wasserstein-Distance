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
