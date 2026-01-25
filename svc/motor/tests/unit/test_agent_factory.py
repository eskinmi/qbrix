import pytest
from unittest.mock import Mock
from unittest.mock import AsyncMock
import numpy as np

from qbrixcore.pool import Pool
from qbrixcore.pool import Arm
from qbrixcore.agent import Agent
from qbrixcore.protoc.stochastic.ts import BetaTSProtocol
from qbrixcore.protoc.stochastic.ts import BetaTSParamState
from qbrixcore.protoc.stochastic.ts import GaussianTSProtocol
from qbrixcore.protoc.stochastic.ucb import UCB1TunedProtocol

from motorsvc.agent_factory import AgentFactory
from motorsvc.agent_factory import PROTOCOL_MAP


class TestProtocolMap:
    def test_protocol_map_contains_beta_ts(self):
        assert "BetaTSProtocol" in PROTOCOL_MAP
        assert PROTOCOL_MAP["BetaTSProtocol"] == BetaTSProtocol

    def test_protocol_map_contains_gaussian_ts(self):
        assert "GaussianTSProtocol" in PROTOCOL_MAP
        assert PROTOCOL_MAP["GaussianTSProtocol"] == GaussianTSProtocol

    def test_protocol_map_contains_ucb1_tuned(self):
        assert "UCB1TunedProtocol" in PROTOCOL_MAP
        assert PROTOCOL_MAP["UCB1TunedProtocol"] == UCB1TunedProtocol


class TestAgentFactoryBuildPool:
    def test_build_pool_creates_pool_with_arms(self, pool_data_dict):
        pool = AgentFactory._build_pool(pool_data_dict)

        assert pool.name == "test-pool"
        assert pool.id == "pool-123"
        assert len(pool.arms) == 3
        assert pool.arms[0].name == "arm-0"
        assert pool.arms[1].name == "arm-1"
        assert pool.arms[2].name == "arm-2"

    def test_build_pool_preserves_arm_ids(self, pool_data_dict):
        pool = AgentFactory._build_pool(pool_data_dict)

        assert pool.arms[0].id == "arm-0"
        assert pool.arms[1].id == "arm-1"
        assert pool.arms[2].id == "arm-2"

    def test_build_pool_sets_arm_active_status(self):
        pool_data = {
            "id": "pool-123",
            "name": "test-pool",
            "arms": [
                {"id": "arm-0", "name": "arm-0", "is_active": True},
                {"id": "arm-1", "name": "arm-1", "is_active": False},
            ],
        }

        pool = AgentFactory._build_pool(pool_data)

        assert pool.arms[0].is_active is True
        assert pool.arms[1].is_active is False

    def test_build_pool_defaults_active_to_true(self):
        pool_data = {
            "id": "pool-123",
            "name": "test-pool",
            "arms": [
                {"id": "arm-0", "name": "arm-0"},  # no is_active field
            ],
        }

        pool = AgentFactory._build_pool(pool_data)

        assert pool.arms[0].is_active is True

    def test_build_pool_with_empty_arms(self):
        pool_data = {"id": "pool-123", "name": "test-pool", "arms": []}

        pool = AgentFactory._build_pool(pool_data)

        assert len(pool.arms) == 0
        assert pool.is_empty


class TestAgentFactoryGetOrCreate:
    @pytest.mark.asyncio
    async def test_get_or_create_returns_cached_agent(
        self, motor_cache, mock_redis_client, mock_agent, beta_ts_params, experiment_data_dict
    ):
        backend = Mock()
        backend.get.return_value = beta_ts_params
        factory = AgentFactory(motor_cache, backend)

        motor_cache.set_agent("exp-123", mock_agent)

        agent = await factory.get_or_create(experiment_data_dict)

        assert agent is not None
        assert agent.experiment_id == "exp-123"
        assert len(agent.pool.arms) == 3

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new_agent_when_not_cached(
        self, motor_cache, mock_redis_client, experiment_data_dict
    ):
        backend = Mock()
        backend.get.return_value = None
        backend.update_params = AsyncMock(return_value=None)
        backend.set = Mock()
        factory = AgentFactory(motor_cache, backend)

        agent = await factory.get_or_create(experiment_data_dict)

        assert agent is not None
        assert agent.experiment_id == "exp-123"
        assert len(agent.pool.arms) == 3
        assert agent.protocol == BetaTSProtocol

    @pytest.mark.asyncio
    async def test_get_or_create_caches_newly_created_agent(
        self, motor_cache, mock_redis_client, experiment_data_dict
    ):
        backend = Mock()
        backend.get.return_value = None
        backend.update_params = AsyncMock(return_value=None)
        backend.set = Mock()
        factory = AgentFactory(motor_cache, backend)

        agent = await factory.get_or_create(experiment_data_dict)

        cached_agent = motor_cache.get_agent("exp-123")
        assert cached_agent is not None
        assert cached_agent.experiment_id == agent.experiment_id

    @pytest.mark.asyncio
    async def test_get_or_create_initializes_params_when_not_in_backend(
        self, motor_cache, mock_redis_client, experiment_data_dict
    ):
        backend = Mock()
        backend.get.return_value = None
        backend.update_params = AsyncMock(return_value=None)
        backend.set = Mock()
        factory = AgentFactory(motor_cache, backend)

        await factory.get_or_create(experiment_data_dict)

        # verify params were initialized and set
        backend.set.assert_called_once()
        call_args = backend.set.call_args
        assert call_args[0][0] == "exp-123"  # experiment_id
        params = call_args[0][1]
        assert params.num_arms == 3

    @pytest.mark.asyncio
    async def test_get_or_create_uses_protocol_params_from_experiment(
        self, motor_cache, mock_redis_client, experiment_data_dict
    ):
        backend = Mock()
        backend.get.return_value = None
        backend.update_params = AsyncMock(return_value=None)
        backend.set = Mock()
        factory = AgentFactory(motor_cache, backend)

        # custom protocol params
        experiment_data_dict["protocol_params"] = {"alpha_prior": 2.0, "beta_prior": 3.0}

        agent = await factory.get_or_create(experiment_data_dict)

        assert agent.init_params == {"alpha_prior": 2.0, "beta_prior": 3.0}

    @pytest.mark.asyncio
    async def test_get_or_create_raises_error_for_unknown_protocol(
        self, motor_cache, mock_redis_client, experiment_data_dict
    ):
        backend = Mock()
        backend.get.return_value = None
        factory = AgentFactory(motor_cache, backend)

        experiment_data_dict["protocol"] = "UnknownProtocol"

        with pytest.raises(ValueError, match="Unknown protocol"):
            await factory.get_or_create(experiment_data_dict)

    @pytest.mark.asyncio
    async def test_get_or_create_fetches_params_from_backend_when_available(
        self, motor_cache, mock_redis_client, experiment_data_dict, beta_ts_params
    ):
        backend = Mock()
        backend.get.return_value = None
        backend.update_params = AsyncMock(return_value=beta_ts_params)
        backend.set = Mock()
        factory = AgentFactory(motor_cache, backend)

        agent = await factory.get_or_create(experiment_data_dict)

        # when update_params returns params, they are already cached by update_params
        # so set should not be called (params are fetched, not initialized)
        backend.update_params.assert_awaited_once()
        backend.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_updates_params_when_agent_cached_but_params_missing(
        self, motor_cache, mock_redis_client, mock_agent, experiment_data_dict, beta_ts_params
    ):
        backend = Mock()
        backend.get.return_value = None  # params not in cache

        async def mock_update_params(exp_id, protocol):
            return beta_ts_params

        backend.update_params = mock_update_params
        factory = AgentFactory(motor_cache, backend)

        motor_cache.set_agent("exp-123", mock_agent)

        agent = await factory.get_or_create(experiment_data_dict)

        assert agent is not None
        # should have returned the cached agent
        assert agent == mock_agent

    @pytest.mark.asyncio
    async def test_get_or_create_initializes_params_when_agent_cached_and_backend_empty(
        self, motor_cache, mock_redis_client, mock_agent, experiment_data_dict
    ):
        backend = Mock()
        backend.get.return_value = None

        async def mock_update_params(exp_id, protocol):
            return None  # no params in redis

        backend.update_params = mock_update_params
        backend.set = Mock()
        factory = AgentFactory(motor_cache, backend)

        motor_cache.set_agent("exp-123", mock_agent)

        agent = await factory.get_or_create(experiment_data_dict)

        # should initialize and set params
        backend.set.assert_called_once()
        call_args = backend.set.call_args
        assert call_args[0][0] == "exp-123"
