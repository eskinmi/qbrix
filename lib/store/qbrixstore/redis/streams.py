import json
from dataclasses import dataclass
from typing import Callable, Awaitable

import redis.asyncio as redis

from qbrixstore.config import RedisSettings


@dataclass
class FeedbackEvent:
    experiment_id: str
    request_id: str
    arm_index: int
    reward: float
    context_id: str
    context_vector: list[float]
    context_metadata: dict
    timestamp_ms: int

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "request_id": self.request_id,
            "arm_index": str(self.arm_index),
            "reward": str(self.reward),
            "context_id": self.context_id,
            "context_vector": json.dumps(self.context_vector),
            "context_metadata": json.dumps(self.context_metadata),
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeedbackEvent":
        return cls(
            experiment_id=data["experiment_id"],
            request_id=data["request_id"],
            arm_index=int(data["arm_index"]),
            reward=float(data["reward"]),
            context_id=data["context_id"],
            context_vector=json.loads(data["context_vector"]),
            context_metadata=json.loads(data["context_metadata"]),
            timestamp_ms=int(data["timestamp_ms"]),
        )


class RedisStreamPublisher:
    def __init__(self, settings: RedisSettings | None = None):
        if settings is None:
            settings = RedisSettings()
        self._settings = settings
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        self._client = redis.from_url(self._settings.url, decode_responses=True)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    async def publish(self, event: FeedbackEvent) -> str:
        if self._client is None:
            raise RuntimeError("Publisher not connected. Call connect() first.")
        message_id = await self._client.xadd(
            self._settings.stream_name, event.to_dict()
        )
        return message_id


class RedisStreamConsumer:
    def __init__(
        self, settings: RedisSettings | None = None, consumer_name: str = "worker-0"
    ):
        if settings is None:
            settings = RedisSettings()
        self._settings = settings
        self._consumer_name = consumer_name
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
    ) -> list[tuple[str, FeedbackEvent]]:
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
                event = FeedbackEvent.from_dict(data)
                events.append((message_id, event))

        return events

    async def ack(self, message_ids: list[str]) -> None:
        if self._client is None:
            raise RuntimeError("Consumer not connected. Call connect() first.")
        if message_ids:
            await self._client.xack(
                self._settings.stream_name, self._settings.consumer_group, *message_ids
            )

    async def run(
        self,
        handler: Callable[[list[FeedbackEvent]], Awaitable[None]],
        batch_size: int = 100,
        block_ms: int = 5000,
    ) -> None:
        while True:
            messages = await self.consume(batch_size=batch_size, block_ms=block_ms)
            if not messages:
                continue

            message_ids = [mid for mid, _ in messages]
            events = [event for _, event in messages]

            await handler(events)
            await self.ack(message_ids)
