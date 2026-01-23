import logging
from typing import Union, List

import grpc

from qbrixproto import common_pb2, motor_pb2, motor_pb2_grpc


logger = logging.getLogger(__name__)


class MotorClient:
    def __init__(self, address: str):
        self._address = address
        self._channel: grpc.aio.Channel | None = None
        self._stub: motor_pb2_grpc.MotorServiceStub | None = None

    async def connect(self) -> None:
        logger.debug('motor client connecting.')
        self._channel = grpc.aio.insecure_channel(self._address)
        self._stub = motor_pb2_grpc.MotorServiceStub(self._channel)
        logger.debug('motor client connected.')

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()

    async def select(
        self,
        experiment_id: str,
        context_id: str,
        context_vector: List[Union[int, float]],
        context_metadata: dict,
    ) -> dict:
        if self._stub is None:
            raise RuntimeError("MotorClient not connected. Call connect() first.")

        request = motor_pb2.SelectRequest(
            experiment_id=experiment_id,
            context=common_pb2.Context(
                id=context_id,
                vector=context_vector,
                metadata={k: str(v) for k, v in context_metadata.items()},
            ),
        )
        response = await self._stub.Select(request)
        return {
            "arm": {
                "id": response.arm.id,
                "name": response.arm.name,
                "index": response.arm.index,
            },
            "request_id": response.request_id,
        }

    async def health(self) -> bool:
        if self._stub is None:
            raise RuntimeError("MotorClient not connected. Call connect() first.")

        request = common_pb2.HealthCheckRequest()
        response = await self._stub.Health(request)
        return response.status == common_pb2.HealthCheckResponse.SERVING
