from __future__ import annotations

from cachebox import TTLCache

from qbrixstore.redis import RedisClient

from proxysvc.gate.config import FeatureGateConfig
from proxysvc.config import ProxySettings


class GateConfigCache:
    """two-level cache for feature gate configurations.

    l1: in-memory TTLCache (microsecond access)
    l2: redis (millisecond access)
    """

    def __init__(self, redis: RedisClient, settings: ProxySettings):
        self._redis = redis
        self._settings = settings
        self._cache: TTLCache = TTLCache(
            maxsize=settings.gate_cache_maxsize, ttl=settings.gate_cache_ttl
        )

    async def get(self, experiment_id: str) -> FeatureGateConfig | None:
        """get gate config from cache hierarchy (l1 -> l2)."""
        if (cached := self._cache.get(experiment_id)) is not None:
            return cached

        data = await self._redis.get_gate_config(experiment_id)
        if data is None:
            return None

        config = FeatureGateConfig.model_validate(data)
        self._cache[experiment_id] = config
        return config

    async def set(self, experiment_id: str, config: FeatureGateConfig) -> None:
        """set gate config in both cache levels."""
        await self._redis.set_gate_config(
            experiment_id=experiment_id,
            config=config.model_dump(mode="json"),
            ttl=self._settings.gate_redis_ttl,
        )
        self._cache[experiment_id] = config

    async def delete(self, experiment_id: str) -> None:
        """delete gate config from both cache levels."""
        await self._redis.delete_gate_config(experiment_id)
        self._cache.pop(experiment_id, None)

    def invalidate(self, experiment_id: str) -> None:
        """invalidate l1 cache entry."""
        self._cache.pop(experiment_id, None)
