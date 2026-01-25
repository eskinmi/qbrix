import pytest
from unittest.mock import AsyncMock
from unittest.mock import Mock
import grpc

from qbrixproto import common_pb2
from qbrixproto import motor_pb2

from motorsvc.server import MotorGRPCServicer
from motorsvc.service import MotorService


class TestMotorGRPCServicerSelect:
    @pytest.mark.asyncio
    async def test_select_calls_service_select(self, motor_settings, mock_grpc_context):
        mock_service = AsyncMock(spec=MotorService)
        mock_service.select.return_value = {
            "arm": {"id": "arm-1", "name": "arm-1", "index": 1}
        }

        servicer = MotorGRPCServicer(mock_service)

        request = motor_pb2.SelectRequest(
            experiment_id="exp-123",
            context=common_pb2.Context(
                id="ctx-1",
                vector=[0.5, 0.3],
                metadata={"user_id": "user-1"},
            ),
        )

        await servicer.Select(request, mock_grpc_context)

        # verify service was called
        mock_service.select.assert_awaited_once()
        call_kwargs = mock_service.select.call_args.kwargs
        assert call_kwargs["experiment_id"] == "exp-123"
        assert call_kwargs["context_id"] == "ctx-1"
        # protobuf floats may have precision issues, verify length and approximate values
        assert len(call_kwargs["context_vector"]) == 2
        assert pytest.approx(call_kwargs["context_vector"][0], rel=1e-6) == 0.5
        assert pytest.approx(call_kwargs["context_vector"][1], rel=1e-6) == 0.3

    @pytest.mark.asyncio
    async def test_select_returns_arm_in_response(self, motor_settings, mock_grpc_context):
        mock_service = AsyncMock(spec=MotorService)
        mock_service.select.return_value = {
            "arm": {"id": "arm-2", "name": "variant-b", "index": 2}
        }

        servicer = MotorGRPCServicer(mock_service)

        request = motor_pb2.SelectRequest(
            experiment_id="exp-123",
            context=common_pb2.Context(id="ctx-1", vector=[], metadata={}),
        )

        response = await servicer.Select(request, mock_grpc_context)

        assert response.arm.id == "arm-2"
        assert response.arm.name == "variant-b"
        assert response.arm.index == 2

    @pytest.mark.asyncio
    async def test_select_handles_value_error_with_not_found_status(
        self, motor_settings, mock_grpc_context
    ):
        mock_service = AsyncMock(spec=MotorService)
        mock_service.select.side_effect = ValueError("experiment not found: exp-999")

        servicer = MotorGRPCServicer(mock_service)

        request = motor_pb2.SelectRequest(
            experiment_id="exp-999",
            context=common_pb2.Context(id="ctx-1", vector=[], metadata={}),
        )

        response = await servicer.Select(request, mock_grpc_context)

        mock_grpc_context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)
        mock_grpc_context.set_details.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_handles_general_exception_with_internal_status(
        self, motor_settings, mock_grpc_context
    ):
        mock_service = AsyncMock(spec=MotorService)
        mock_service.select.side_effect = Exception("unexpected error")

        servicer = MotorGRPCServicer(mock_service)

        request = motor_pb2.SelectRequest(
            experiment_id="exp-123",
            context=common_pb2.Context(id="ctx-1", vector=[], metadata={}),
        )

        response = await servicer.Select(request, mock_grpc_context)

        mock_grpc_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
        mock_grpc_context.set_details.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_returns_empty_response_on_error(self, motor_settings, mock_grpc_context):
        mock_service = AsyncMock(spec=MotorService)
        mock_service.select.side_effect = ValueError("experiment not found")

        servicer = MotorGRPCServicer(mock_service)

        request = motor_pb2.SelectRequest(
            experiment_id="exp-999",
            context=common_pb2.Context(id="ctx-1", vector=[], metadata={}),
        )

        response = await servicer.Select(request, mock_grpc_context)

        # empty response
        assert response == motor_pb2.SelectResponse()

    @pytest.mark.asyncio
    async def test_select_handles_empty_context_vector(self, motor_settings, mock_grpc_context):
        mock_service = AsyncMock(spec=MotorService)
        mock_service.select.return_value = {
            "arm": {"id": "arm-0", "name": "arm-0", "index": 0}
        }

        servicer = MotorGRPCServicer(mock_service)

        request = motor_pb2.SelectRequest(
            experiment_id="exp-123",
            context=common_pb2.Context(id="ctx-1", vector=[], metadata={}),
        )

        await servicer.Select(request, mock_grpc_context)

        mock_service.select.assert_awaited_once()
        call_args = mock_service.select.call_args
        assert call_args.kwargs["context_vector"] == []

    @pytest.mark.asyncio
    async def test_select_converts_protobuf_repeated_to_list(self, motor_settings, mock_grpc_context):
        mock_service = AsyncMock(spec=MotorService)
        mock_service.select.return_value = {
            "arm": {"id": "arm-0", "name": "arm-0", "index": 0}
        }

        servicer = MotorGRPCServicer(mock_service)

        request = motor_pb2.SelectRequest(
            experiment_id="exp-123",
            context=common_pb2.Context(
                id="ctx-1",
                vector=[0.1, 0.2, 0.3],
                metadata={},
            ),
        )

        await servicer.Select(request, mock_grpc_context)

        call_kwargs = mock_service.select.call_args.kwargs
        # verify list conversion
        assert isinstance(call_kwargs["context_vector"], list)
        assert len(call_kwargs["context_vector"]) == 3


class TestMotorGRPCServicerHealth:
    @pytest.mark.asyncio
    async def test_health_returns_serving_when_healthy(self, motor_settings, mock_grpc_context):
        mock_service = AsyncMock(spec=MotorService)
        mock_service.health.return_value = True

        servicer = MotorGRPCServicer(mock_service)

        request = common_pb2.HealthCheckRequest()

        response = await servicer.Health(request, mock_grpc_context)

        assert response.status == common_pb2.HealthCheckResponse.SERVING

    @pytest.mark.asyncio
    async def test_health_returns_not_serving_when_unhealthy(
        self, motor_settings, mock_grpc_context
    ):
        mock_service = AsyncMock(spec=MotorService)
        mock_service.health.return_value = False

        servicer = MotorGRPCServicer(mock_service)

        request = common_pb2.HealthCheckRequest()

        response = await servicer.Health(request, mock_grpc_context)

        assert response.status == common_pb2.HealthCheckResponse.NOT_SERVING

    @pytest.mark.asyncio
    async def test_health_calls_service_health(self, motor_settings, mock_grpc_context):
        mock_service = AsyncMock(spec=MotorService)
        mock_service.health.return_value = True

        servicer = MotorGRPCServicer(mock_service)

        request = common_pb2.HealthCheckRequest()

        await servicer.Health(request, mock_grpc_context)

        mock_service.health.assert_awaited_once()
