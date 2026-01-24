from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection

from qbrixlog import get_logger
from qbrixproto import common_pb2, motor_pb2_grpc
from qbrixproto import motor_pb2

from motorsvc.config import MotorSettings
from motorsvc.service import MotorService

logger = get_logger(__name__)


class MotorGRPCServicer(motor_pb2_grpc.MotorServiceServicer):
    def __init__(self, service: MotorService):
        self._service = service

    async def Select(self, request, context):
        try:
            result = await self._service.select(
                experiment_id=request.experiment_id,
                context_id=request.context.id,
                context_vector=list(request.context.vector),
                context_metadata=dict(request.context.metadata),
            )
            return motor_pb2.SelectResponse(
                arm=common_pb2.Arm(
                    id=result["arm"]["id"],
                    name=result["arm"]["name"],
                    index=result["arm"]["index"],
                ),
            )
        except ValueError as e:
            logger.warning("experiment not found: %s", request.experiment_id)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return motor_pb2.SelectResponse()
        except Exception as e:
            logger.error("selection failed for experiment %s: %s", request.experiment_id, e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return motor_pb2.SelectResponse()

    async def Health(self, request, context):
        healthy = await self._service.health()
        return common_pb2.HealthCheckResponse(
            status=(
                common_pb2.HealthCheckResponse.SERVING
                if healthy
                else common_pb2.HealthCheckResponse.NOT_SERVING
            )
        )


async def serve(settings: MotorSettings | None = None) -> None:
    if settings is None:
        settings = MotorSettings()

    service = MotorService(settings)
    await service.start()

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    motor_pb2_grpc.add_MotorServiceServicer_to_server(
        MotorGRPCServicer(service), server
    )

    service_names = (
        motor_pb2.DESCRIPTOR.services_by_name["MotorService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)

    logger.info("starting motor grpc server on %s", listen_addr)
    await server.start()

    try:
        await server.wait_for_termination()
    finally:
        logger.info("shutting down motor grpc server")
        await service.stop()
        await server.stop(grace=5)
