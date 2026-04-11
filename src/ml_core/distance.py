import numpy as np
from scipy.stats import wasserstein_distance


def compute_wasserstein_distance(dist1, dist2):
    """
    Compute Wasserstein (Earth Mover's) distance between two distributions.

    Parameters:
    dist1, dist2 : array-like
        Input distributions

    Returns:
    float
        Wasserstein distance
    """
    # print(dist1, dist2)
    return wasserstein_distance(dist1, dist2)