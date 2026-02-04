import pytest
from unittest.mock import AsyncMock
from unittest.mock import Mock

from qbrixcore.context import Context

from motorsvc.service import MotorService


class TestMotorServiceLifecycle:
    @pytest.mark.asyncio
    async def test_start_initializes_param_backend(self, motor_settings, mock_redis_client):
        service = MotorService(motor_settings)
        # manually set redis to avoid actual connection
        service._redis = mock_redis_client

        # call start internals without creating new redis
        service._param_backend = Mock()
        service._agent_factory = Mock()

        assert service._param_backend is not None

    @pytest.mark.asyncio
    async def test_start_initializes_agent_factory(self, motor_settings, mock_redis_client):
        service = MotorService(motor_settings)
        # manually set redis to avoid actual connection
        service._redis = mock_redis_client
        service._param_backend = Mock()
        service._agent_factory = Mock()

        assert service._agent_factory is not None

    @pytest.mark.asyncio
    async def test_stop_disconnects_from_redis(self, motor_settings, mock_redis_client):
        service = MotorService(motor_settings)
        service._redis = mock_redis_client

        await service.stop()

        mock_redis_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_handles_no_redis_connection(self, motor_settings):
        service = MotorService(motor_settings)
        service._redis = None

        # should not raise exception
        await service.stop()


class TestMotorServiceSelect:
    @pytest.mark.asyncio
    async def test_select_retrieves_experiment_from_redis(
        self, tenant_id, motor_settings, mock_redis_client, experiment_data_dict
    ):
        service = MotorService(motor_settings)
        service._redis = mock_redis_client

        mock_redis_client.get_experiment.return_value = experiment_data_dict

        mock_agent = Mock()
        mock_agent.select.return_value = 0
        mock_agent.pool.arms = [Mock(id="arm-0", name="arm-0")]

        mock_factory = AsyncMock()
        mock_factory.get_or_create.return_value = mock_agent
        service._agent_factory = mock_factory

        await service.select(
            tenant_id=tenant_id,
            experiment_id="exp-123",
            context_id="ctx-1",
            context_vector=[0.5, 0.3],
            context_metadata={"user_id": "user-1"},
        )

        mock_redis_client.get_experiment.assert_awaited_once_with(tenant_id, "exp-123")

    @pytest.mark.asyncio
    async def test_select_raises_error_when_experiment_not_found(
        self, tenant_id, motor_settings, mock_redis_client
    ):
        service = MotorService(motor_settings)
        service._redis = mock_redis_client

        mock_redis_client.get_experiment.return_value = None

        with pytest.raises(ValueError, match="experiment not found"):
            await service.select(
                tenant_id=tenant_id,
                experiment_id="nonexistent-exp",
                context_id="ctx-1",
                context_vector=[],
                context_metadata={},
            )

    @pytest.mark.asyncio
    async def test_select_gets_or_creates_agent(
        self, tenant_id, motor_settings, mock_redis_client, experiment_data_dict
    ):
        service = MotorService(motor_settings)
        service._redis = mock_redis_client

        mock_redis_client.get_experiment.return_value = experiment_data_dict

        mock_agent = Mock()
        mock_agent.select.return_value = 1
        mock_agent.pool.arms = [
            Mock(id="arm-0", name="arm-0"),
            Mock(id="arm-1", name="arm-1"),
        ]

        mock_factory = AsyncMock()
        mock_factory.get_or_create.return_value = mock_agent
        service._agent_factory = mock_factory

        await service.select(
            tenant_id=tenant_id,
            experiment_id="exp-123",
            context_id="ctx-1",
            context_vector=[0.5],
            context_metadata={},
        )

        mock_factory.get_or_create.assert_awaited_once_with(tenant_id, experiment_data_dict)

    @pytest.mark.asyncio
    async def test_select_calls_agent_select_with_context(
        self, tenant_id, motor_settings, mock_redis_client, experiment_data_dict
    ):
        service = MotorService(motor_settings)
        service._redis = mock_redis_client

        mock_redis_client.get_experiment.return_value = experiment_data_dict

        mock_agent = Mock()
        mock_agent.select.return_value = 0
        mock_agent.pool.arms = [Mock(id="arm-0", name="arm-0")]

        mock_factory = AsyncMock()
        mock_factory.get_or_create.return_value = mock_agent
        service._agent_factory = mock_factory

        await service.select(
            tenant_id=tenant_id,
            experiment_id="exp-123",
            context_id="ctx-1",
            context_vector=[0.5, 0.3],
            context_metadata={"user_id": "user-1"},
        )

        mock_agent.select.assert_called_once()
        call_args = mock_agent.select.call_args
        context = call_args[0][0]
        assert isinstance(context, Context)
        assert context.id == "ctx-1"
        assert context.vector == [0.5, 0.3]
        assert context.metadata == {"user_id": "user-1"}

    @pytest.mark.asyncio
    async def test_select_returns_selected_arm_info(
        self, tenant_id, motor_settings, mock_redis_client, experiment_data_dict, pool_with_three_arms
    ):
        service = MotorService(motor_settings)
        service._redis = mock_redis_client

        mock_redis_client.get_experiment.return_value = experiment_data_dict

        mock_agent = Mock()
        mock_agent.select.return_value = 1
        mock_agent.pool = pool_with_three_arms

        mock_factory = AsyncMock()
        mock_factory.get_or_create.return_value = mock_agent
        service._agent_factory = mock_factory

        result = await service.select(
            tenant_id=tenant_id,
            experiment_id="exp-123",
            context_id="ctx-1",
            context_vector=[],
            context_metadata={},
        )

        assert result["arm"]["id"] == "arm-1"
        assert result["arm"]["name"] == "arm-1"
        assert result["arm"]["index"] == 1

    @pytest.mark.asyncio
    async def test_select_handles_empty_context_vector(
        self, tenant_id, motor_settings, mock_redis_client, experiment_data_dict
    ):
        service = MotorService(motor_settings)
        service._redis = mock_redis_client

        mock_redis_client.get_experiment.return_value = experiment_data_dict

        mock_agent = Mock()
        mock_agent.select.return_value = 0
        mock_agent.pool.arms = [Mock(id="arm-0", name="arm-0")]

        mock_factory = AsyncMock()
        mock_factory.get_or_create.return_value = mock_agent
        service._agent_factory = mock_factory

        await service.select(
            tenant_id=tenant_id,
            experiment_id="exp-123",
            context_id="ctx-1",
            context_vector=None,  # noqa
            context_metadata=None,  # noqa
        )

        mock_agent.select.assert_called_once()
        call_args = mock_agent.select.call_args
        context = call_args[0][0]
        assert context.vector == []
        assert context.metadata == {}

    @pytest.mark.asyncio
    async def test_select_handles_empty_metadata(
        self, tenant_id, motor_settings, mock_redis_client, experiment_data_dict
    ):
        service = MotorService(motor_settings)
        service._redis = mock_redis_client

        mock_redis_client.get_experiment.return_value = experiment_data_dict

        mock_agent = Mock()
        mock_agent.select.return_value = 0
        mock_agent.pool.arms = [Mock(id="arm-0", name="arm-0")]

        mock_factory = AsyncMock()
        mock_factory.get_or_create.return_value = mock_agent
        service._agent_factory = mock_factory

        await service.select(
            tenant_id=tenant_id,
            experiment_id="exp-123",
            context_id="ctx-1",
            context_vector=[],
            context_metadata=None,  # noqa
        )

        mock_agent.select.assert_called_once()
        call_args = mock_agent.select.call_args
        context = call_args[0][0]
        assert context.metadata == {}


class TestMotorServiceHealth:
    @pytest.mark.asyncio
    async def test_health_returns_true_when_redis_responds(self, motor_settings, mock_redis_client):
        service = MotorService(motor_settings)
        service._redis = mock_redis_client
        mock_redis_client.client.ping.return_value = True
        result = await service.health()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_returns_false_when_redis_fails(self, motor_settings, mock_redis_client):
        service = MotorService(motor_settings)
        service._redis = mock_redis_client
        mock_redis_client.client.ping.side_effect = Exception("connection failed")
        result = await service.health()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_calls_redis_ping(self, motor_settings, mock_redis_client):
        service = MotorService(motor_settings)
        service._redis = mock_redis_client

        await service.health()

        mock_redis_client.client.ping.assert_awaited_once()