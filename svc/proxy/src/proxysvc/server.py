from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection

from qbrixproto import common_pb2, proxy_pb2, proxy_pb2_grpc

from proxysvc.config import ProxySettings
from proxysvc.service import ProxyService
from proxysvc.token import TokenError, TokenExpiredError, TokenInvalidError


class ProxyGRPCServicer(proxy_pb2_grpc.ProxyServiceServicer):
    def __init__(self, service: ProxyService):
        self._service = service

    async def CreatePool(self, request, context):
        try:
            arms = [{"name": arm.name, "metadata": dict(arm.metadata) if hasattr(arm, 'metadata') else {}} for arm in request.arms]
            response = await self._service.create_pool(request.name, arms)
            return proxy_pb2.CreatePoolResponse(
                pool=self._dict_to_pool(response)
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return proxy_pb2.CreatePoolResponse()

    async def GetPool(self, request, context):
        response = await self._service.get_pool(request.pool_id)
        if response is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Pool not found: {request.pool_id}")
            return proxy_pb2.GetPoolResponse()
        return proxy_pb2.GetPoolResponse(
            pool=self._dict_to_pool(response)
        )

    async def DeletePool(self, request, context):
        deleted = await self._service.delete_pool(request.pool_id)
        if not deleted:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Pool not found: {request.pool_id}")
        return proxy_pb2.DeletePoolResponse(deleted=deleted)

    async def CreateExperiment(self, request, context):
        try:
            feature_gate = None
            if request.HasField("feature_gate"):
                feature_gate = self._proto_to_gate_config(request.feature_gate)
            response = await self._service.create_experiment(
                name=request.name,
                pool_id=request.pool_id,
                protocol=request.protocol,
                protocol_params=dict(request.protocol_params),
                enabled=request.enabled,
                feature_gate_config=feature_gate
            )
            return proxy_pb2.CreateExperimentResponse(
                experiment=self._dict_to_experiment(response)
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return proxy_pb2.CreateExperimentResponse()

    async def GetExperiment(self, request, context):
        response = await self._service.get_experiment(request.experiment_id)
        if response is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Experiment not found: {request.experiment_id}")
            return proxy_pb2.GetExperimentResponse()
        return proxy_pb2.GetExperimentResponse(
            experiment=self._dict_to_experiment(response)
        )

    async def UpdateExperiment(self, request, context):
        try:
            kwargs = {}
            if request.HasField("enabled"):
                kwargs["enabled"] = request.enabled
            if request.protocol_params:
                kwargs["protocol_params"] = dict(request.protocol_params)
            if request.HasField("feature_gate"):
                kwargs["feature_gate_config"] = self._proto_to_gate_config(request.feature_gate)
            response = await self._service.update_experiment(request.experiment_id, **kwargs)
            if response is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Experiment not found: {request.experiment_id}")
                return proxy_pb2.UpdateExperimentResponse()
            return proxy_pb2.UpdateExperimentResponse(
                experiment=self._dict_to_experiment(response)
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return proxy_pb2.UpdateExperimentResponse()

    async def DeleteExperiment(self, request, context):
        deleted = await self._service.delete_experiment(request.experiment_id)
        if not deleted:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Experiment not found: {request.experiment_id}")
        return proxy_pb2.DeleteExperimentResponse(deleted=deleted)

    async def CreateGateConfig(self, request, context):
        try:
            config = self._proto_to_gate_config(request.config)
            response = await self._service.create_gate_config(request.experiment_id, config)
            if response is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Experiment not found: {request.experiment_id}")
                return proxy_pb2.CreateGateConfigResponse()
            return proxy_pb2.CreateGateConfigResponse(
                config=self._dict_to_gate_config(response)
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return proxy_pb2.CreateGateConfigResponse()

    async def GetGateConfig(self, request, context):
        response = await self._service.get_gate_config(request.experiment_id)
        if response is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Gate config not found for experiment: {request.experiment_id}")
            return proxy_pb2.GetGateConfigResponse()
        return proxy_pb2.GetGateConfigResponse(
            config=self._dict_to_gate_config(response)
        )

    async def UpdateGateConfig(self, request, context):
        try:
            config = self._proto_to_gate_config(request.config)
            response = await self._service.update_gate_config(request.experiment_id, config)
            if response is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Gate config not found for experiment: {request.experiment_id}")
                return proxy_pb2.UpdateGateConfigResponse()
            return proxy_pb2.UpdateGateConfigResponse(
                config=self._dict_to_gate_config(response)
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return proxy_pb2.UpdateGateConfigResponse()

    async def DeleteGateConfig(self, request, context):
        deleted = await self._service.delete_gate_config(request.experiment_id)
        if not deleted:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Gate config not found for experiment: {request.experiment_id}")
        return proxy_pb2.DeleteGateConfigResponse(deleted=deleted)

    async def Select(self, request, context):
        try:
            response = await self._service.select(
                experiment_id=request.experiment_id,
                context_id=request.context.id,
                context_vector=list(request.context.vector),
                context_metadata=dict(request.context.metadata)
            )
            return proxy_pb2.SelectResponse(
                arm=common_pb2.Arm(
                    id=response["arm"]["id"],
                    name=response["arm"]["name"],
                    index=response["arm"]["index"]
                ),
                request_id=response["request_id"],
                is_default=response.get("is_default", False)
            )
        except ValueError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return proxy_pb2.SelectResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return proxy_pb2.SelectResponse()

    async def Feedback(self, request, context):
        if not request.request_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("request_id is required")
            return proxy_pb2.FeedbackResponse(accepted=False)

        try:
            accepted = await self._service.feed(
                request_id=request.request_id,
                reward=request.reward,
            )
            return proxy_pb2.FeedbackResponse(accepted=accepted)
        except TokenExpiredError as e:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(str(e))
            return proxy_pb2.FeedbackResponse(accepted=False)
        except TokenInvalidError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return proxy_pb2.FeedbackResponse(accepted=False)
        except TokenError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return proxy_pb2.FeedbackResponse(accepted=False)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return proxy_pb2.FeedbackResponse(accepted=False)

    async def Health(self, request, context):
        healthy = await self._service.health()
        return common_pb2.HealthCheckResponse(
            status=common_pb2.HealthCheckResponse.SERVING if healthy
            else common_pb2.HealthCheckResponse.NOT_SERVING
        )

    @staticmethod
    def _dict_to_pool(d: dict) -> common_pb2.Pool:
        return common_pb2.Pool(
            id=d["id"],
            name=d["name"],
            arms=[
                common_pb2.Arm(id=arm["id"], name=arm["name"], index=arm["index"])
                for arm in d.get("arms", [])
            ]
        )

    @staticmethod
    def _dict_to_experiment(d: dict) -> common_pb2.Experiment:
        return common_pb2.Experiment(
            id=d["id"],
            name=d["name"],
            pool_id=d["pool_id"],
            protocol=d["protocol"],
            protocol_params={k: str(v) for k, v in d.get("protocol_params", {}).items()},
            enabled=d.get("enabled", False)
        )

    @staticmethod
    def _proto_to_gate_config(proto: proxy_pb2.FeatureGateConfig) -> dict:
        """convert proto FeatureGateConfig to dict for service layer."""
        from datetime import datetime, time

        config = {
            "enabled": proto.enabled,
            "rollout_percentage": proto.rollout_percentage,
            "default_arm_id": proto.default_arm_id or None,
            "timezone": proto.timezone or "UTC",
            "rules": []
        }

        # parse schedule
        if proto.HasField("schedule"):
            if proto.schedule.start_timestamp_ms:
                config["schedule_start"] = datetime.fromtimestamp(
                    proto.schedule.start_timestamp_ms / 1000
                )
            if proto.schedule.end_timestamp_ms:
                config["schedule_end"] = datetime.fromtimestamp(
                    proto.schedule.end_timestamp_ms / 1000
                )

        # parse active hours
        if proto.HasField("active_hours"):
            if proto.active_hours.start:
                h, m = map(int, proto.active_hours.start.split(":"))
                config["active_hours_start"] = time(h, m)
            if proto.active_hours.end:
                h, m = map(int, proto.active_hours.end.split(":"))
                config["active_hours_end"] = time(h, m)

        # parse rules
        import json
        for rule in proto.rules:
            try:
                value = json.loads(rule.value)
            except (json.JSONDecodeError, TypeError):
                value = rule.value
            config["rules"].append({
                "key": rule.key,
                "operator": rule.operator,
                "value": value,
                "arm": {"committed": {"id": rule.arm_id}}
            })

        return config

    @staticmethod
    def _dict_to_gate_config(d: dict) -> proxy_pb2.FeatureGateConfig:
        """convert dict from service layer to proto FeatureGateConfig."""
        import json

        experiment = d.get("experiment", {})
        schedule = experiment.get("schedule", {})
        period = schedule.get("period", {})
        hour = schedule.get("hour", {})
        rollout = experiment.get("rollout", {})

        proto_config = proxy_pb2.FeatureGateConfig(
            enabled=experiment.get("enabled", True),
            rollout_percentage=rollout.get("percentage", 100.0),
            default_arm_id=experiment.get("arm", {}).get("committed", {}).get("id") or "",
            timezone=hour.get("timezone") or period.get("timezone") or "UTC"
        )

        # set schedule
        if period.get("start"):
            start_dt = period["start"]
            if isinstance(start_dt, str):
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_dt)
            proto_config.schedule.start_timestamp_ms = int(start_dt.timestamp() * 1000)
        if period.get("end"):
            end_dt = period["end"]
            if isinstance(end_dt, str):
                from datetime import datetime
                end_dt = datetime.fromisoformat(end_dt)
            proto_config.schedule.end_timestamp_ms = int(end_dt.timestamp() * 1000)

        # set active hours
        if hour.get("start"):
            start_time = hour["start"]
            if hasattr(start_time, "strftime"):
                proto_config.active_hours.start = start_time.strftime("%H:%M")
            else:
                proto_config.active_hours.start = str(start_time)
        if hour.get("end"):
            end_time = hour["end"]
            if hasattr(end_time, "strftime"):
                proto_config.active_hours.end = end_time.strftime("%H:%M")
            else:
                proto_config.active_hours.end = str(end_time)

        # set rules
        for rule in d.get("rules", []):
            proto_config.rules.append(proxy_pb2.RuleConfig(
                key=rule.get("key", ""),
                operator=rule.get("operator", ""),
                value=json.dumps(rule.get("value")) if not isinstance(rule.get("value"), str) else rule.get("value", ""),
                arm_id=rule.get("arm", {}).get("committed", {}).get("id") or ""
            ))

        return proto_config


async def serve(settings: ProxySettings | None = None) -> None:
    if settings is None:
        settings = ProxySettings()

    service = ProxyService(settings)
    await service.start()

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    proxy_pb2_grpc.add_ProxyServiceServicer_to_server(ProxyGRPCServicer(service), server)

    service_names = (
        proxy_pb2.DESCRIPTOR.services_by_name['ProxyService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)

    print(f"Starting Proxy gRPC server on {listen_addr}")
    await server.start()

    try:
        await server.wait_for_termination()
    finally:
        await service.stop()
        await server.stop(grace=5)
