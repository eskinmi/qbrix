from qbrixcore.param.backend import BaseParamBackend
from qbrixcore.param.state import BaseParamState
from qbrixcore.protoc.base import BaseProtocol
from qbrixstore.redis.client import RedisClient

from motorsvc.cache import MotorCache


class RedisBackedInMemoryParamBackend:
    """param backend that supports multi-tenant operations."""

    def __init__(self, redis_client: RedisClient, cache: MotorCache):
        self._redis = redis_client
        self._cache = cache

    def get(self, tenant_id: str, experiment_id: str) -> BaseParamState | None:
        return self._cache.get_params(tenant_id, experiment_id)

    def set(self, tenant_id: str, experiment_id: str, params: BaseParamState) -> None:
        self._cache.set_params(tenant_id, experiment_id, params)

    async def update_params(
        self, tenant_id: str, experiment_id: str, protocol: type[BaseProtocol]
    ) -> BaseParamState | None:
        params_dict = await self._redis.get_params(tenant_id, experiment_id)
        if params_dict is not None:
            params = protocol.param_state_cls.model_validate(params_dict)
            self._cache.set_params(tenant_id, experiment_id, params)
            return params
        return None

    def scoped(self, tenant_id: str) -> "TenantScopedParamBackend":
        """create a tenant-scoped param backend for use with Agent."""
        return TenantScopedParamBackend(self, tenant_id)


class TenantScopedParamBackend(BaseParamBackend):
    """param backend scoped to a specific tenant for Agent compatibility."""

    def __init__(self, parent: RedisBackedInMemoryParamBackend, tenant_id: str):
        self._parent = parent
        self._tenant_id = tenant_id

    def get(self, experiment_id: str) -> BaseParamState | None:
        return self._parent.get(self._tenant_id, experiment_id)

    def set(self, experiment_id: str, params: BaseParamState) -> None:
        self._parent.set(self._tenant_id, experiment_id, params)
