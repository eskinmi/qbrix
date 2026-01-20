from __future__ import annotations

import uuid
from dataclasses import dataclass

import grpc

from qbrixproto import common_pb2
from qbrixproto import proxy_pb2
from qbrixproto import proxy_pb2_grpc

from .config import settings


@dataclass
class SelectResult:
    request_id: str
    arm_id: str
    arm_index: int
    is_default: bool


class ProxyClient:
    """grpc client wrapper for load testing."""

    def __init__(self, address: str | None = None):
        self.address = address or settings.proxy_address
        self._channel: grpc.Channel | None = None
        self._stub: proxy_pb2_grpc.ProxyServiceStub | None = None

    def connect(self) -> None:
        self._channel = grpc.insecure_channel(self.address)
        self._stub = proxy_pb2_grpc.ProxyServiceStub(self._channel)

    def close(self) -> None:
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None

    @property
    def is_connected(self) -> bool:
        return self._stub is not None

    @property
    def stub(self) -> proxy_pb2_grpc.ProxyServiceStub:
        if self._stub is None:
            raise RuntimeError("client not connected, call connect() first")
        return self._stub

    def health_check(self) -> bool:
        request = common_pb2.HealthCheckRequest()
        response = self.stub.Health(request)
        return response.status == common_pb2.HealthCheckResponse.SERVING

    def create_pool(self, name: str, num_arms: int) -> str:
        arms = [common_pb2.Arm(name=f"arm-{i}", index=i) for i in range(num_arms)]
        request = proxy_pb2.CreatePoolRequest(name=name, arms=arms)
        response = self.stub.CreatePool(request)
        return response.pool.id

    def create_experiment(
        self,
        name: str,
        pool_id: str,
        protocol: str = "beta_ts",
        enabled: bool = True,
    ) -> str:
        request = proxy_pb2.CreateExperimentRequest(
            name=name,
            pool_id=pool_id,
            protocol=protocol,
            enabled=enabled,
        )
        response = self.stub.CreateExperiment(request)
        return response.experiment.id

    def delete_pool(self, pool_id: str) -> bool:
        request = proxy_pb2.DeletePoolRequest(pool_id=pool_id)
        response = self.stub.DeletePool(request)
        return response.deleted

    def delete_experiment(self, experiment_id: str) -> bool:
        request = proxy_pb2.DeleteExperimentRequest(experiment_id=experiment_id)
        response = self.stub.DeleteExperiment(request)
        return response.deleted

    def select(
        self,
        experiment_id: str,
        context_id: str | None = None,
        context_vector: list[float] | None = None,
        context_metadata: dict[str, str] | None = None,
    ) -> SelectResult:
        context = common_pb2.Context(
            id=context_id or str(uuid.uuid4()),
            vector=context_vector or [],
            metadata=context_metadata or {},
        )
        request = proxy_pb2.SelectRequest(
            experiment_id=experiment_id,
            context=context,
        )
        response = self.stub.Select(request)
        return SelectResult(
            request_id=response.request_id,
            arm_id=response.arm.id,
            arm_index=response.arm.index,
            is_default=response.is_default,
        )

    def feedback(self, request_id: str, reward: float) -> bool:
        request = proxy_pb2.FeedbackRequest(
            request_id=request_id,
            reward=reward,
        )
        response = self.stub.Feedback(request)
        return response.accepted
