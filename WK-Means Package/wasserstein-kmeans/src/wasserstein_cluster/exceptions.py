"""Custom exceptions and warnings for the wasserstein-cluster library."""


class WassersteinClusterError(Exception):
    """Base exception for all errors in the wasserstein-cluster library."""


class ConvergenceWarning(UserWarning):
    """Warning raised when an algorithm fails to converge within max_iter."""


class EmptyClusterWarning(UserWarning):
    """Warning raised when a cluster loses all member points during an iteration."""
