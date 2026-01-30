from cachebox import TTLCache

from qbrixcore.param.state import BaseParamState

from motorsvc.config import MotorSettings


class MotorCache:
    def __init__(self, settings: MotorSettings):
        self._settings = settings
        self._params: TTLCache = TTLCache(
            maxsize=settings.param_cache_maxsize, ttl=settings.param_cache_ttl
        )
        self._agents: TTLCache = TTLCache(
            maxsize=settings.agent_cache_maxsize, ttl=settings.agent_cache_ttl
        )

    @staticmethod
    def _cache_key(tenant_id: str, experiment_id: str) -> str:
        return f"{tenant_id}:{experiment_id}"

    def get_params(self, tenant_id: str, experiment_id: str) -> BaseParamState | None:
        return self._params.get(self._cache_key(tenant_id, experiment_id))

    def set_params(self, tenant_id: str, experiment_id: str, params: BaseParamState) -> None:
        self._params[self._cache_key(tenant_id, experiment_id)] = params

    def get_agent(self, tenant_id: str, experiment_id: str):
        return self._agents.get(self._cache_key(tenant_id, experiment_id))

    def set_agent(self, tenant_id: str, experiment_id: str, agent) -> None:
        self._agents[self._cache_key(tenant_id, experiment_id)] = agent

    def invalidate_experiment(self, tenant_id: str, experiment_id: str) -> None:
        cache_key = self._cache_key(tenant_id, experiment_id)
        self._params.pop(cache_key, None)
        self._agents.pop(cache_key, None)

    def clear(self) -> None:
        self._params.clear()
        self._agents.clear()
