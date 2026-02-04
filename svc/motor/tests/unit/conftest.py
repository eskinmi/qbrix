import pytest
from unittest.mock import Mock
from unittest.mock import AsyncMock
import numpy as np

from qbrixcore.pool import Pool
from qbrixcore.pool import Arm
from qbrixcore.agent import Agent
from qbrixcore.policy.stochastic.ts import BetaTSPolicy
from qbrixcore.policy.stochastic.ts import BetaTSParamState
from qbrixcore.policy.stochastic.ts import GaussianTSParamState
from qbrixstore.redis.client import RedisClient

from motorsvc.config import MotorSettings
from motorsvc.cache import MotorCache


@pytest.fixture
def tenant_id():
    """default tenant id for testing"""
    return "tenant-001"


@pytest.fixture
def motor_settings():
    """default motor settings for testing"""
    return MotorSettings(
        grpc_host="localhost",
        grpc_port=50051,
        redis_host="localhost",
        redis_port=6379,
        param_cache_ttl=60,
        param_cache_maxsize=100,
        agent_cache_ttl=300,
        agent_cache_maxsize=50,
    )


@pytest.fixture
def motor_cache(motor_settings):
    """motor cache instance with test settings"""
    return MotorCache(motor_settings)


@pytest.fixture
def mock_redis_client():
    """mocked redis client"""
    mock = AsyncMock(spec=RedisClient)
    mock.client = AsyncMock()
    mock.client.ping = AsyncMock(return_value=True)
    mock.connect = AsyncMock()
    mock.close = AsyncMock()
    mock.get_experiment = AsyncMock()
    mock.get_params = AsyncMock()
    return mock


@pytest.fixture
def pool_with_three_arms():
    """pool with three arms for testing"""
    pool = Pool(name="test-pool", id="pool-123")
    pool.add_arm(Arm(name="arm-0", id="arm-0"))
    pool.add_arm(Arm(name="arm-1", id="arm-1"))
    pool.add_arm(Arm(name="arm-2", id="arm-2"))
    return pool


@pytest.fixture
def pool_data_dict():
    """pool data as dict (experiment record format)"""
    return {
        "id": "pool-123",
        "name": "test-pool",
        "arms": [
            {"id": "arm-0", "name": "arm-0", "is_active": True},
            {"id": "arm-1", "name": "arm-1", "is_active": True},
            {"id": "arm-2", "name": "arm-2", "is_active": True},
        ],
    }


@pytest.fixture
def experiment_data_dict(pool_data_dict):
    """experiment data as dict (redis record format)"""
    return {
        "id": "exp-123",
        "name": "test-experiment",
        "policy": "BetaTSPolicy",
        "policy_params": {"alpha_prior": 1.0, "beta_prior": 1.0},
        "pool": pool_data_dict,
    }


@pytest.fixture
def beta_ts_params():
    """initialized beta ts param state"""
    return BetaTSParamState(
        num_arms=3,
        alpha=np.array([2.0, 1.5, 1.0]),
        beta=np.array([1.0, 1.5, 2.0]),
        T=np.array([1, 1, 1]),
    )


@pytest.fixture
def gaussian_ts_params():
    """initialized gaussian ts param state"""
    return GaussianTSParamState(
        num_arms=3,
        posterior_mean=np.array([0.5, 0.3, 0.7]),
        posterior_precision=np.array([2.0, 2.0, 2.0]),
        T=np.array([5, 5, 5]),
    )


@pytest.fixture
def mock_agent(pool_with_three_arms, beta_ts_params):
    """mocked agent with beta ts policy"""
    mock_backend = Mock()
    mock_backend.get.return_value = beta_ts_params

    agent = Agent(
        experiment_id="exp-123",
        pool=pool_with_three_arms,
        policy=BetaTSPolicy,
        init_params={"alpha_prior": 1.0, "beta_prior": 1.0},
        param_backend=mock_backend,
    )
    return agent


@pytest.fixture
def mock_grpc_context():
    """mocked grpc context for servicer tests"""
    context = Mock()
    context.set_code = Mock()
    context.set_details = Mock()
    return context
