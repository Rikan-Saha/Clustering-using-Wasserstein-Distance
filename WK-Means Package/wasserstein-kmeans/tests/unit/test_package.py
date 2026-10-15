"""Sanity test for package import and version."""

import wasserstein_cluster


def test_package_metadata() -> None:
    assert wasserstein_cluster.__version__ == "0.1.0"
