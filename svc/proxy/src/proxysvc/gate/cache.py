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

    @staticmethod
    def _cache_key(tenant_id: str, experiment_id: str) -> str:
        return f"{tenant_id}:{experiment_id}"

    async def get(self, tenant_id: str, experiment_id: str) -> FeatureGateConfig | None:
        """get gate config from cache hierarchy (l1 -> l2)."""
        cache_key = self._cache_key(tenant_id, experiment_id)
        if (cached := self._cache.get(cache_key)) is not None:
            return cached

        data = await self._redis.get_gate_config(tenant_id, experiment_id)
        if data is None:
            return None

        config = FeatureGateConfig.model_validate(data)
        self._cache[cache_key] = config
        return config

    async def set(self, tenant_id: str, experiment_id: str, config: FeatureGateConfig) -> None:
        """set gate config in both cache levels."""
        await self._redis.set_gate_config(
            tenant_id=tenant_id,
            experiment_id=experiment_id,
            config=config.model_dump(mode="json"),
            ttl=self._settings.gate_redis_ttl,
        )
        cache_key = self._cache_key(tenant_id, experiment_id)
        self._cache[cache_key] = config

    async def delete(self, tenant_id: str, experiment_id: str) -> None:
        """delete gate config from both cache levels."""
        await self._redis.delete_gate_config(tenant_id, experiment_id)
        cache_key = self._cache_key(tenant_id, experiment_id)
        self._cache.pop(cache_key, None)

    def invalidate(self, tenant_id: str, experiment_id: str) -> None:
        """invalidate l1 cache entry."""
        cache_key = self._cache_key(tenant_id, experiment_id)
        self._cache.pop(cache_key, None)
