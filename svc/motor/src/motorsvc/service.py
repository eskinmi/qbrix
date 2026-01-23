import uuid

from qbrixcore.context import Context
from qbrixstore.redis.client import RedisClient
from qbrixstore.config import RedisSettings

from motorsvc.cache import MotorCache
from motorsvc.config import MotorSettings
from motorsvc.param_backend import RedisBackedInMemoryParamBackend
from motorsvc.agent_factory import AgentFactory


class MotorService:
    def __init__(self, settings: MotorSettings):
        self._settings = settings
        self._cache = MotorCache(settings)
        self._redis: RedisClient | None = None
        self._param_backend: RedisBackedInMemoryParamBackend | None = None
        self._agent_factory: AgentFactory | None = None

    async def start(self) -> None:
        redis_settings = RedisSettings(
            host=self._settings.redis_host,
            port=self._settings.redis_port,
            password=self._settings.redis_password,
            db=self._settings.redis_db,
        )
        self._redis = RedisClient(redis_settings)
        await self._redis.connect()
        self._param_backend = RedisBackedInMemoryParamBackend(self._redis, self._cache)
        self._agent_factory = AgentFactory(self._cache, self._param_backend)

    async def stop(self) -> None:
        if self._redis:
            await self._redis.close()

    async def select(
        self,
        experiment_id: str,
        context_id: str,
        context_vector: list[float],
        context_metadata: dict,
    ) -> dict:
        experiment_data = await self._redis.get_experiment(experiment_id)
        if experiment_data is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        agent = await self._agent_factory.get_or_create(experiment_data)

        context = Context(
            id=context_id, vector=context_vector or [], metadata=context_metadata or {}
        )

        choice_index = agent.select(context)
        arm = agent.pool.arms[choice_index]

        return {
            "arm": {"id": arm.id, "name": arm.name, "index": choice_index},
            "request_id": uuid.uuid4().hex,
        }

    async def health(self) -> bool:
        try:
            await self._redis.client.ping()
            return True
        except Exception:  # noqa
            return False
