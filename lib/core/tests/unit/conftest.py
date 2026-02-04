"""Shared fixtures for qbrixcore unit tests."""

import numpy as np
import pytest

from qbrixcore.pool import Pool
from qbrixcore.pool import Arm
from qbrixcore.context import Context
from qbrixcore.param.backend import InMemoryParamBackend


@pytest.fixture
def pool_with_three_arms():
    """Creates a pool with three active arms."""
    pool = Pool(name="test-pool", id="pool-123")
    pool.add_arm(Arm(name="arm-0", id="arm-0"))
    pool.add_arm(Arm(name="arm-1", id="arm-1"))
    pool.add_arm(Arm(name="arm-2", id="arm-2"))
    return pool


@pytest.fixture
def pool_with_five_arms():
    """Creates a pool with five active arms."""
    pool = Pool(name="test-pool-5", id="pool-456")
    for i in range(5):
        pool.add_arm(Arm(name=f"arm-{i}", id=f"arm-{i}"))
    return pool


@pytest.fixture
def empty_pool():
    """Creates an empty pool with no arms."""
    return Pool(name="empty-pool", id="pool-empty")


@pytest.fixture
def context_without_vector():
    """Creates a context without a feature vector (for stochastic policies)."""
    return Context(id="ctx-123")


@pytest.fixture
def context_with_vector():
    """Creates a context with a 5-dimensional feature vector (for contextual policies)."""
    return Context(id="ctx-456", vector=[1.0, 0.5, -0.3, 0.8, 0.2])


@pytest.fixture
def context_with_numpy_vector():
    """Creates a context with numpy array vector."""
    return Context(id="ctx-789", vector=np.array([1.0, 0.5, -0.3, 0.8, 0.2]))


@pytest.fixture
def in_memory_backend():
    """Creates an in-memory parameter backend."""
    return InMemoryParamBackend()


@pytest.fixture
def experiment_id():
    """Returns a standard experiment ID for testing."""
    return "exp-test-123"
