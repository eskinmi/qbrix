import json

import redis.asyncio as redis

from qbrixstore.config import RedisSettings


class RedisClient:
    def __init__(self, settings: RedisSettings | None = None):
        if settings is None:
            settings = RedisSettings()
        self._settings = settings
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        # Attention: Uses a single async connection, not a connection pool.
        # This is sufficient because: (1) async I/O multiplexes concurrent requests,
        # (2) motorsvc uses local TTL caching to reduce Redis calls on hot path,
        # (3) services scale horizontally. Consider ConnectionPool if contention occurs.
        self._client = redis.from_url(self._settings.url, decode_responses=True)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        return self._client

    @staticmethod
    def _param_key(experiment_id: str) -> str:
        return f"qbrix:params:{experiment_id}"

    @staticmethod
    def _experiment_key(experiment_id: str) -> str:
        return f"qbrix:experiment:{experiment_id}"

    @staticmethod
    def _gate_key(experiment_id: str) -> str:
        return f"qbrix:gate:{experiment_id}"

    async def get_params(self, experiment_id: str) -> dict | None:
        data = await self.client.get(self._param_key(experiment_id))
        if data is None:
            return None
        return json.loads(data)

    async def set_params(
        self, experiment_id: str, params: dict, ttl: int | None = None
    ) -> None:
        key = self._param_key(experiment_id)
        await self.client.set(key, json.dumps(params), ex=ttl)

    async def get_experiment(self, experiment_id: str) -> dict | None:
        data = await self.client.get(self._experiment_key(experiment_id))
        if data is None:
            return None
        return json.loads(data)

    async def set_experiment(
        self, experiment_id: str, experiment: dict, ttl: int | None = None
    ) -> None:
        key = self._experiment_key(experiment_id)
        await self.client.set(key, json.dumps(experiment), ex=ttl)

    async def delete_experiment(self, experiment_id: str) -> None:
        await self.client.delete(
            self._experiment_key(experiment_id), self._param_key(experiment_id)
        )

    async def get_gate_config(self, experiment_id: str) -> dict | None:
        data = await self.client.get(self._gate_key(experiment_id))
        if data is None:
            return None
        return json.loads(data)

    async def set_gate_config(
        self, experiment_id: str, config: dict, ttl: int | None = None
    ) -> None:
        key = self._gate_key(experiment_id)
        await self.client.set(key, json.dumps(config), ex=ttl)

    async def delete_gate_config(self, experiment_id: str) -> None:
        await self.client.delete(self._gate_key(experiment_id))

    async def publish_gate_invalidation(self, channel: str, experiment_id: str) -> None:
        """publish gate config invalidation event for cache listeners."""
        await self.client.publish(channel, experiment_id)
