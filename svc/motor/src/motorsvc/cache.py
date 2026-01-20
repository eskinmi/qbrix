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

    def get_params(self, experiment_id: str) -> BaseParamState | None:
        return self._params.get(experiment_id)

    def set_params(self, experiment_id: str, params: BaseParamState) -> None:
        self._params[experiment_id] = params

    def get_agent(self, experiment_id: str):
        return self._agents.get(experiment_id)

    def set_agent(self, experiment_id: str, agent) -> None:
        self._agents[experiment_id] = agent

    def invalidate_experiment(self, experiment_id: str) -> None:
        self._params.pop(experiment_id, None)
        self._agents.pop(experiment_id, None)

    def clear(self) -> None:
        self._params.clear()
        self._agents.clear()
