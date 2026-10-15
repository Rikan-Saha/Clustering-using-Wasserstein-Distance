import numpy as np
import os
import pandas as pd

def generate_synthetic_distributions(n_samples=10, size=100):
    data = []
    for i in range(n_samples):
        mean = np.random.uniform(-2, 2)
        std = np.random.uniform(0.5, 1.5)
        dist = np.random.normal(mean, std, size)
        data.append(dist)
    return data


def save_data(data, filename="synthetic_data.csv"):
    """
    Save list of distributions to CSV file
    """

    os.makedirs("../data", exist_ok=True)
    filepath = os.path.join("../data", filename)

    # Convert list of arrays → DataFrame
    df = pd.DataFrame(data)

    df.to_csv(filepath, index=False)

    print(f"Data saved to {filepath}")