"""Distribution preprocessing, empirical CDF estimators, and quantile transformers."""

from wasserstein_cluster.preprocessing.distributions import (
    empirical_cdf,
    quantile_resample,
    resample_collection,
    standardize_distribution,
)

__all__ = [
    "empirical_cdf",
    "quantile_resample",
    "standardize_distribution",
    "resample_collection",
]
