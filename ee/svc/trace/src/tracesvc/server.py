import asyncio
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection

from qbrixlog import get_logger
from qbrixproto import common_pb2

from tracesvc.config import TraceSettings
from tracesvc.service import TraceService

logger = get_logger(__name__)


class TraceGRPCServicer:
    """grpc servicer for trace service health and stats."""

    def __init__(self, service: TraceService):
        self._service = service

    async def Health(self, request, context):  # noqa
        healthy = await self._service.health()
        return common_pb2.HealthCheckResponse(
            status=(
                common_pb2.HealthCheckResponse.SERVING
                if healthy
                else common_pb2.HealthCheckResponse.NOT_SERVING
            )
        )


async def serve(settings: TraceSettings | None = None) -> None:

    if settings is None:
        settings = TraceSettings()

    service = TraceService(settings)
    await service.start()

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))

    service_names = (
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)

    logger.info("starting trace grpc server on %s", listen_addr)
    await server.start()

    selection_task = asyncio.create_task(service.run_selection_consumer())
    feedback_task = asyncio.create_task(service.run_feedback_consumer())

    try:
        await server.wait_for_termination()
    finally:
        logger.info("shutting down trace grpc server")
        selection_task.cancel()
        feedback_task.cancel()
        await service.stop()
        await server.stop(grace=5)
