import asyncio
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection

from qbrixproto import common_pb2, cortex_pb2, cortex_pb2_grpc

from cortexsvc.config import CortexSettings
from cortexsvc.service import CortexService


class CortexGRPCServicer(cortex_pb2_grpc.CortexServiceServicer):
    def __init__(self, service: CortexService):
        self._service = service

    async def FlushBatch(self, request, context):
        try:
            count = await self._service.flush_batch(request.experiment_id or None)
            return cortex_pb2.FlushBatchResponse(events_processed=count)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cortex_pb2.FlushBatchResponse(events_processed=0)

    async def GetStats(self, request, context):
        stats_list = self._service.get_stats(request.experiment_id or None)
        return cortex_pb2.GetStatsResponse(
            stats=[
                cortex_pb2.ExperimentStats(
                    experiment_id=s["experiment_id"],
                    total_events=s.get("total", 0),
                    pending_events=s.get("pending", 0),
                    last_train_timestamp_ms=s.get("last_train", 0)
                )
                for s in stats_list
            ]
        )

    async def Health(self, request, context):
        healthy = await self._service.health()
        return common_pb2.HealthCheckResponse(
            status=common_pb2.HealthCheckResponse.SERVING if healthy
            else common_pb2.HealthCheckResponse.NOT_SERVING
        )


async def serve(settings: CortexSettings | None = None) -> None:
    if settings is None:
        settings = CortexSettings()

    service = CortexService(settings)
    await service.start()

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    cortex_pb2_grpc.add_CortexServiceServicer_to_server(CortexGRPCServicer(service), server)

    service_names = (
        cortex_pb2.DESCRIPTOR.services_by_name['CortexService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)

    print(f"Starting Cortex gRPC server on {listen_addr}")
    await server.start()

    consumer_task = asyncio.create_task(service.run_consumer())

    try:
        await server.wait_for_termination()
    finally:
        consumer_task.cancel()
        await service.stop()
        await server.stop(grace=5)
