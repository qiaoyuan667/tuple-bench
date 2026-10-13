from pathlib import Path

import pytest

from tuplebench import Benchmark


@pytest.fixture(scope="session")
def release_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def benchmark(release_root):
    return Benchmark(release_root)
