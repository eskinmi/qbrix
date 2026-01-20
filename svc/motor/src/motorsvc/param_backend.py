from qbrixcore.param.backend import BaseParamBackend
from qbrixcore.param.state import BaseParamState
from qbrixcore.protoc.base import BaseProtocol
from qbrixstore.redis.client import RedisClient

from motorsvc.cache import MotorCache


class RedisBackedInMemoryParamBackend(BaseParamBackend):
    def __init__(self, redis_client: RedisClient, cache: MotorCache):
        self._redis = redis_client
        self._cache = cache

    def get(self, experiment_id: str) -> BaseParamState | None:
        return self._cache.get_params(experiment_id)

    def set(self, experiment_id: str, params: BaseParamState) -> None:
        self._cache.set_params(experiment_id, params)

    async def update_params(
        self, experiment_id: str, protocol: type[BaseProtocol]
    ) -> BaseParamState | None:
        params_dict = await self._redis.get_params(experiment_id)
        if params_dict is not None:
            params = protocol.param_state_cls.model_validate(params_dict)
            self._cache.set_params(experiment_id, params)
            return params
        return None
