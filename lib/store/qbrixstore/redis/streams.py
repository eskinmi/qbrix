import json
from dataclasses import dataclass
from typing import Callable, Awaitable, Generic, TypeVar

import redis.asyncio as redis

from qbrixstore.config import RedisSettings


@dataclass
class FeedbackEvent:
    tenant_id: str
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
            "tenant_id": self.tenant_id,
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
            tenant_id=data["tenant_id"],
            experiment_id=data["experiment_id"],
            request_id=data["request_id"],
            arm_index=int(data["arm_index"]),
            reward=float(data["reward"]),
            context_id=data["context_id"],
            context_vector=json.loads(data["context_vector"]),
            context_metadata=json.loads(data["context_metadata"]),
            timestamp_ms=int(data["timestamp_ms"]),
        )


@dataclass
class SelectionEvent:
    """event emitted when an arm is selected for a context."""

    tenant_id: str
    experiment_id: str
    request_id: str
    arm_id: str
    arm_name: str
    arm_index: int
    is_default: bool
    context_id: str
    context_vector: list[float]
    context_metadata: dict
    timestamp_ms: int
    policy: str

    def to_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "experiment_id": self.experiment_id,
            "request_id": self.request_id,
            "arm_id": self.arm_id,
            "arm_name": self.arm_name,
            "arm_index": str(self.arm_index),
            "is_default": "1" if self.is_default else "0",
            "context_id": self.context_id,
            "context_vector": json.dumps(self.context_vector),
            "context_metadata": json.dumps(self.context_metadata),
            "timestamp_ms": str(self.timestamp_ms),
            "policy": self.policy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SelectionEvent":
        return cls(
            tenant_id=data["tenant_id"],
            experiment_id=data["experiment_id"],
            request_id=data["request_id"],
            arm_id=data["arm_id"],
            arm_name=data["arm_name"],
            arm_index=int(data["arm_index"]),
            is_default=data["is_default"] == "1",
            context_id=data["context_id"],
            context_vector=json.loads(data["context_vector"]),
            context_metadata=json.loads(data["context_metadata"]),
            timestamp_ms=int(data["timestamp_ms"]),
            policy=data["policy"],
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

    async def publish(self, event: FeedbackEvent | SelectionEvent) -> str:
        if self._client is None:
            raise RuntimeError("Publisher not connected. Call connect() first.")
        message_id = await self._client.xadd(
            self._settings.stream_name,
            event.to_dict(),
            maxlen=self._settings.stream_max_len,
            approximate=True,
        )
        return message_id


EventT = TypeVar("EventT", FeedbackEvent, SelectionEvent)


class RedisStreamConsumer(Generic[EventT]):
    def __init__(
        self,
        settings: RedisSettings | None = None,
        consumer_name: str = "worker-0",
        event_class: type[EventT] | None = None,
    ):
        if settings is None:
            settings = RedisSettings()
        self._settings = settings
        self._consumer_name = consumer_name
        self._event_class: type = event_class or FeedbackEvent
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
    ) -> list[tuple[str, EventT]]:
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
    ) -> list[tuple[str, EventT]]:
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

        # xautoclaim returns: [next_start_id, [[id, data], ...], [deleted_ids]]
        if not results or len(results) < 2:
            return []

        messages = results[1]
        events = []
        for message_id, data in messages:
            if data:
                event = self._event_class.from_dict(data)
                events.append((message_id, event))

        return events

    async def run(
        self,
        handler: Callable[[list[EventT]], Awaitable[None]],
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
