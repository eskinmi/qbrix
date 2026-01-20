from qbrixcore.pool import Pool, Arm
from qbrixcore.agent import Agent
from qbrixcore.protoc import BaseProtocol

from motorsvc.cache import MotorCache
from motorsvc.param_backend import RedisBackedInMemoryParamBackend


def _build_protocol_map() -> dict[str, type[BaseProtocol]]:
    registry = {}

    def collect(cls):
        for subclass in cls.__subclasses__():
            if hasattr(subclass, "name") and subclass.name:
                registry[subclass.name] = subclass
            collect(subclass)

    collect(BaseProtocol)
    return registry


PROTOCOL_MAP = _build_protocol_map()


class AgentFactory:
    def __init__(
        self, cache: MotorCache, param_backend: RedisBackedInMemoryParamBackend
    ):
        self._cache = cache
        self._param_backend = param_backend

    @staticmethod
    def _build_pool(pool_data: dict) -> Pool:
        pool = Pool(name=pool_data["name"], id=pool_data["id"])
        for arm_data in pool_data["arms"]:
            arm = Arm(
                name=arm_data["name"],
                id=arm_data["id"],
                is_active=arm_data.get("is_active", True),
            )
            pool.add_arm(arm)
        return pool

    async def get_or_create(self, experiment_data: dict) -> Agent:
        """
        Get cached agent or create new one.

        note: this method has an intentional race window between cache check
        and cache set. concurrent requests may build duplicate agents. this is
        acceptable because agents are stateless and param state lives in redis.
        """
        experiment_id = experiment_data["id"]

        agent = self._cache.get_agent(experiment_id)
        if agent is not None:
            if (params := self._param_backend.get(experiment_id)) is None:
                params = await self._param_backend.update_params(
                    experiment_id, agent.protocol
                )
            if params is None:
                params = agent.protocol.init_params(
                    num_arms=len(agent.pool), **agent.init_params
                )
                self._param_backend.set(experiment_id, params)
            return agent

        # if there is no agent, it's either because:
        # 1. it's the first request for an experiment
        # 2. there is a new replica / or instance restarted
        # 3. agent cache is expired / invalidated

        # in all cases we need to regenerate the agent, meaning we need to fetch the protocol, pool, etc.
        # if it is not the first request, we already must have parameters, so we will fetch the
        # parameters from the cache or redis.

        protocol_name = experiment_data["protocol"]
        protocol_cls = PROTOCOL_MAP.get(protocol_name)
        if protocol_cls is None:
            raise ValueError(
                f"Unknown protocol: {protocol_name}. Available: {list(PROTOCOL_MAP.keys())}"
            )

        if self._param_backend.get(experiment_id) is None:
            params = await self._param_backend.update_params(
                experiment_id, protocol_cls
            )
            if params is None:
                params = protocol_cls.init_params(
                    num_arms=len(experiment_data["pool"]["arms"]),
                    **experiment_data.get("protocol_params", {}),
                )
                self._param_backend.set(experiment_id, params)

        pool = self._build_pool(experiment_data["pool"])

        agent = Agent(
            experiment_id=experiment_id,
            pool=pool,
            protocol=protocol_cls,
            init_params=experiment_data.get("protocol_params", {}),
            param_backend=self._param_backend,
        )

        self._cache.set_agent(experiment_id, agent)
        return agent
