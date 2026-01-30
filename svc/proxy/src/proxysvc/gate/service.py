from __future__ import annotations

import logging

from qbrixstore.redis import RedisClient

from proxysvc.config import ProxySettings
from proxysvc.gate.cache import GateConfigCache
from proxysvc.gate.config import FeatureGateConfig
from proxysvc.gate.controller import FeatureGate
from proxysvc.gate.model.base import BaseArmModel

logger = logging.getLogger(__name__)


class GateService:
    """feature gate service for experiment targeting.

    provides fast gate evaluation on the hot path with
    two-level caching and safe fallbacks.
    """

    def __init__(self, redis: RedisClient, settings: ProxySettings):
        self._cache = GateConfigCache(redis, settings)
        self._redis = redis
        self._settings = settings

    async def evaluate(
        self, tenant_id: str, experiment_id: str, context_id: str, context_metadata: dict
    ) -> BaseArmModel | None:
        """evaluate feature gate for a select request.

        returns:
            BaseArmModel: if gate determines a committed arm (skip bandit)
            None: if bandit selection should proceed

        safety: returns None on any error (fail-open to bandit path)
        """
        try:
            config = await self._cache.get(tenant_id, experiment_id)
            if config is None:
                return None

            return FeatureGate.control(config, context_id, context_metadata)

        except Exception as e:
            logger.warning(
                f"gate evaluation failed for {experiment_id}, falling back to bandit: {e}"
            )
            return None

    async def get_config(self, tenant_id: str, experiment_id: str) -> FeatureGateConfig | None:
        """get gate config from cache."""
        return await self._cache.get(tenant_id, experiment_id)

    async def set_config(self, tenant_id: str, experiment_id: str, config: FeatureGateConfig) -> None:
        """set gate config in cache."""
        await self._cache.set(tenant_id, experiment_id, config)

    async def delete_config(self, tenant_id: str, experiment_id: str) -> None:
        """delete gate config from cache."""
        await self._cache.delete(tenant_id, experiment_id)

    def invalidate(self, tenant_id: str, experiment_id: str) -> None:
        """invalidate l1 cache entry."""
        self._cache.invalidate(tenant_id, experiment_id)
