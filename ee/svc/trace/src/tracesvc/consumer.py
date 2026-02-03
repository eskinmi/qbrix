from __future__ import annotations

from typing import TypeVar, Generic, Type

import redis.asyncio as redis

from qbrixstore.config import RedisSettings

T = TypeVar("T")


class GenericStreamConsumer(Generic[T]):
    """generic redis stream consumer that works with any event type with from_dict method."""

    def __init__(
        self,
        settings: RedisSettings,
        consumer_name: str,
        event_class: Type[T],
    ):
        self._settings = settings
        self._consumer_name = consumer_name
        self._event_class = event_class
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        self._client = redis.from_url(self._settings.url, decode_responses=True)
        try:
            await self._client.xgroup_create(
                self._settings.stream_name,
                self._settings.consumer_group,
                id="0",
                mkstream=True,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    async def consume(
        self, batch_size: int = 100, block_ms: int = 5000
    ) -> list[tuple[str, T]]:
        if self._client is None:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        results = await self._client.xreadgroup(
            groupname=self._settings.consumer_group,
            consumername=self._consumer_name,
            streams={self._settings.stream_name: ">"},
            count=batch_size,
            block=block_ms,
        )

        events = []
        for stream_name, messages in results:
            for message_id, data in messages:
                event = self._event_class.from_dict(data)
                events.append((message_id, event))

        return events

    async def ack(self, message_ids: list[str]) -> None:
        if self._client is None:
            raise RuntimeError("Consumer not connected. Call connect() first.")
        if message_ids:
            await self._client.xack(
                self._settings.stream_name, self._settings.consumer_group, *message_ids
            )
            await self._client.xdel(self._settings.stream_name, *message_ids)

    async def get_pending_count(self) -> int:
        """get count of pending messages for this consumer."""
        if self._client is None:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        info = await self._client.xpending(
            self._settings.stream_name,
            self._settings.consumer_group,
        )

        if not info or info["pending"] == 0:
            return 0

        for consumer_info in info.get("consumers", []):
            if consumer_info["name"] == self._consumer_name:
                return consumer_info["pending"]

        return 0

    async def claim_pending(
        self, count: int = 100, min_idle_ms: int = 0
    ) -> list[tuple[str, T]]:
        """claim pending messages that were read but not acked (e.g., after crash)."""
        if self._client is None:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        results = await self._client.xautoclaim(
            self._settings.stream_name,
            self._settings.consumer_group,
            self._consumer_name,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=count,
        )

        if not results or len(results) < 2:
            return []

        messages = results[1]
        events = []
        for message_id, data in messages:
            if data:
                event = self._event_class.from_dict(data)
                events.append((message_id, event))

        return events
