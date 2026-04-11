# 📊 Wasserstein KMeans Clustering & Analytics System

## 🚀 Project Overview

This project builds an end-to-end machine learning system for clustering distribution-based data using Wasserstein KMeans.

Unlike traditional KMeans that uses Euclidean distance, this system uses Wasserstein distance to cluster probability distributions, making it suitable for complex structured data such as distributions and time-series-like samples.

The system integrates:
- Python (ML implementation)
- FastAPI (model serving)
- Jupyter Notebooks (experimentation)
- Bokeh (interactive visualization)

---

## 🎯 Business Objective

To design a clustering system capable of grouping distribution-based data samples accurately using optimal transport theory (Wasserstein distance), enabling better pattern discovery in complex datasets.

---

## 🏗️ Project Architecture
```
Data Generation / Input
        ↓
Synthetic Data Creation (Python)
        ↓
Wasserstein Distance Computation
        ↓
KMeans Clustering Algorithm
        ↓
Cluster Assignment
        ↓
FastAPI Inference Layer
        ↓
Interactive Visualization (Bokeh)
```
---

## 📂 Project Structure
```
Project6/
│
├── data/
│   └── synthetic_data.csv
│
├── notebooks/
│   └── experimentation.ipynb
│
├── api/
│   ├── main.py
│   ├── service.py
│   ├── schemas.py
│   ├── path_setup.py
│
├── src/
│   └── ml_core/
│       ├── kmeans_wasserstein.py
│       ├── distance.py
│       ├── utils.py
│       ├── visualization.py
│
├── README.md
```
---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

git clone https://github.com/your-username/wasserstein-kmeans.git
cd wasserstein-kmeans

---

### 2️⃣ Create Virtual Environment

python -m venv env
env\Scripts\activate   # Windows

---

### 3️⃣ Install Dependencies

pip install fastapi uvicorn numpy pandas bokeh pydantic

---

## ▶️ How to Run the Project

### Step 1: Run FastAPI Server

uvicorn api.main:app --reload

---

### Step 2: Open API Docs

http://127.0.0.1:8000/docs

---

### Step 3: Example API Input

{
  "data": [
    [1.0, 1.2, 0.9, 1.1],
    [5.0, 5.1, 4.9, 5.2],
    [0.5, 0.6, 0.4, 0.55]
  ],
  "n_clusters": 2
}

---

### ✅ Expected Output

{
  "clusters": [0, 1, 0]
}

---

## 📊 Visualization

from ml_core.visualization import plot_clusters_interactive

plot_clusters_interactive(data, clusters)

Features:
- Interactive histograms
- Cluster-wise comparison
- Zoom and hover support

---

## 🧠 Key Concepts Used

- Wasserstein Distance (Optimal Transport)
- K-Means Clustering
- Distribution-based similarity learning
- Synthetic data generation
- FastAPI deployment

---

## 🔥 Key Highlights

- Custom ML algorithm from scratch
- Real-world distribution clustering
- Production-style API design
- Interactive visualization layer
- Modular architecture

---

## 💼 Business Impact

- Useful for probabilistic data clustering
- Applicable in finance and risk modeling
- Helps in pattern discovery in complex datasets
- Demonstrates ML system design capability

---

## 🚀 Future Enhancements

- GPU acceleration for Wasserstein distance
- Streamlit dashboard UI
- Docker deployment
- Model persistence (pickle/joblib)
- Scalable inference pipeline

---

## 🙏 Acknowledgment

Thank you for reviewing this project.