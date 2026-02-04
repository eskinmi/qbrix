from qbrixcore.pool import Pool, Arm
from qbrixcore.agent import Agent
from qbrixcore.policy import BasePolicy

from motorsvc.cache import MotorCache
from motorsvc.param_backend import RedisBackedInMemoryParamBackend


def _build_policy_map() -> dict[str, type[BasePolicy]]:
    registry = {}

    def collect(cls):
        for subclass in cls.__subclasses__():
            if hasattr(subclass, "name") and subclass.name:
                registry[subclass.name] = subclass
            collect(subclass)

    collect(BasePolicy)
    return registry


PROTOCOL_MAP = _build_policy_map()


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

    async def get_or_create(self, tenant_id: str, experiment_record: dict) -> Agent:
        """
        Get cached agent or create new one.

        note: this method has an intentional race window between cache check
        and cache set. concurrent requests may build duplicate agents. this is
        acceptable because agents are stateless and param state lives in redis.
        """
        experiment_id = experiment_record["id"]

        agent = self._cache.get_agent(tenant_id, experiment_id)
        if agent is not None:
            if (params := self._param_backend.get(tenant_id, experiment_id)) is None:
                params = await self._param_backend.update_params(
                    tenant_id, experiment_id, agent.policy
                )
            if params is None:
                params = agent.policy.init_params(
                    num_arms=len(agent.pool), **agent.init_params
                )
                self._param_backend.set(tenant_id, experiment_id, params)
            return agent

        # attention:
        #  if there is no agent, it's either because:
        #  1. it's the first request for an experiment
        #  2. there is a new replica / or instance restarted
        #  3. agent cache is expired / invalidated
        #  <->
        #  in all cases we need to regenerate the agent, meaning we need to fetch the policy, pool, etc.
        #  if it is not the first request, we already must have parameters, so we will fetch the
        #  parameters from the cache or redis.

        policy_name = experiment_record["policy"]
        policy_cls = PROTOCOL_MAP.get(policy_name)
        if policy_cls is None:
            raise ValueError(
                f"Unknown policy: {policy_name}. Available: {list(PROTOCOL_MAP.keys())}"
            )

        if self._param_backend.get(tenant_id, experiment_id) is None:
            params = await self._param_backend.update_params(
                tenant_id, experiment_id, policy_cls
            )
            if params is None:
                params = policy_cls.init_params(
                    num_arms=len(experiment_record["pool"]["arms"]),
                    **experiment_record.get("policy_params", {}),
                )
                self._param_backend.set(tenant_id, experiment_id, params)

        pool = self._build_pool(experiment_record["pool"])
        scoped_param_backend = self._param_backend.scoped(tenant_id)
        agent = Agent(
            experiment_id=experiment_id,
            pool=pool,
            policy=policy_cls,
            init_params=experiment_record.get("policy_params", {}),
            param_backend=scoped_param_backend,
        )

        self._cache.set_agent(tenant_id, experiment_id, agent)
        return agent
