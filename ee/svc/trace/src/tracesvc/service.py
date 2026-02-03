from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from qbrixlog import get_logger
from qbrixstore.clickhouse.client import ClickHouseClient
from qbrixstore.clickhouse.migrations import create_tables
from qbrixstore.redis.streams import SelectionEvent
from qbrixstore.redis.streams import FeedbackEvent
from qbrixstore.config import RedisSettings
from qbrixstore.config import ClickHouseSettings

from tracesvc.config import TraceSettings
from tracesvc.consumer import GenericStreamConsumer

logger = get_logger(__name__)


class TraceService:
    """service for persisting selection and feedback events to clickhouse."""

    def __init__(self, settings: TraceSettings):
        self._settings = settings
        self._clickhouse: ClickHouseClient | None = None
        self._selection_consumer: GenericStreamConsumer[SelectionEvent] | None = None
        self._feedback_consumer: GenericStreamConsumer[FeedbackEvent] | None = None
        self._stats: dict[str, dict] = defaultdict(
            lambda: {"selections": 0, "feedback": 0, "last_write": 0}
        )
        self._running = False

        self._selection_pending: list[tuple[str, SelectionEvent]] = []
        self._feedback_pending: list[tuple[str, FeedbackEvent]] = []
        self._selection_lock = asyncio.Lock()
        self._feedback_lock = asyncio.Lock()

    async def start(self) -> None:
        clickhouse_settings = ClickHouseSettings(
            host=self._settings.clickhouse_host,
            port=self._settings.clickhouse_port,
            user=self._settings.clickhouse_user,
            password=self._settings.clickhouse_password,
            database=self._settings.clickhouse_database,
        )
        self._clickhouse = ClickHouseClient(clickhouse_settings)
        self._clickhouse.connect()
        logger.info(
            "connected to clickhouse at %s:%s",
            self._settings.clickhouse_host,
            self._settings.clickhouse_port,
        )

        create_tables(self._clickhouse.client)
        logger.info("clickhouse tables initialized")

        selection_redis_settings = RedisSettings(
            host=self._settings.redis_host,
            port=self._settings.redis_port,
            password=self._settings.redis_password,
            db=self._settings.redis_db,
            stream_name=self._settings.selection_stream_name,
            consumer_group=self._settings.consumer_group,
        )
        self._selection_consumer = GenericStreamConsumer(
            settings=selection_redis_settings,
            consumer_name=self._settings.consumer_name,
            event_class=SelectionEvent,
        )
        await self._selection_consumer.connect()
        logger.info("selection stream consumer started: %s", self._settings.consumer_name)

        feedback_redis_settings = RedisSettings(
            host=self._settings.redis_host,
            port=self._settings.redis_port,
            password=self._settings.redis_password,
            db=self._settings.redis_db,
            stream_name=self._settings.feedback_stream_name,
            consumer_group=self._settings.consumer_group,
        )
        self._feedback_consumer = GenericStreamConsumer(
            settings=feedback_redis_settings,
            consumer_name=self._settings.consumer_name,
            event_class=FeedbackEvent,
        )
        await self._feedback_consumer.connect()
        logger.info("feedback stream consumer started: %s", self._settings.consumer_name)

        self._running = True

    async def stop(self) -> None:
        self._running = False

        async with self._selection_lock:
            if self._selection_pending:
                logger.info("flushing %d pending selection events", len(self._selection_pending))
                try:
                    await self._process_selection_batch(self._selection_pending)
                    self._selection_pending = []
                except Exception as e:  # noqa
                    logger.error("failed to flush selection events: %s", e)

        async with self._feedback_lock:
            if self._feedback_pending:
                logger.info("flushing %d pending feedback events", len(self._feedback_pending))
                try:
                    await self._process_feedback_batch(self._feedback_pending)
                    self._feedback_pending = []
                except Exception as e:  # noqa
                    logger.error("failed to flush feedback events: %s", e)

        if self._selection_consumer:
            await self._selection_consumer.close()
        if self._feedback_consumer:
            await self._feedback_consumer.close()
        if self._clickhouse:
            self._clickhouse.close()

        logger.info("trace service stopped")

    async def _process_selection_batch(
        self, messages: list[tuple[str, SelectionEvent]]
    ) -> None:
        """process a batch of selection events."""
        message_ids = [mid for mid, _ in messages]
        events = [event for _, event in messages]

        self._clickhouse.insert_selection_events(events)

        for event in events:
            stats_key = f"{event.tenant_id}:{event.experiment_id}"
            self._stats[stats_key]["selections"] += 1
            self._stats[stats_key]["last_write"] = int(time.time() * 1000)

        await self._selection_consumer.ack(message_ids)
        logger.info("persisted %d selection events", len(events))

    async def _process_feedback_batch(
        self, messages: list[tuple[str, FeedbackEvent]]
    ) -> None:
        """process a batch of feedback events."""
        message_ids = [mid for mid, _ in messages]
        events = [event for _, event in messages]

        self._clickhouse.insert_feedback_events(events)

        for event in events:
            stats_key = f"{event.tenant_id}:{event.experiment_id}"
            self._stats[stats_key]["feedback"] += 1
            self._stats[stats_key]["last_write"] = int(time.time() * 1000)

        await self._feedback_consumer.ack(message_ids)
        logger.info("persisted %d feedback events", len(events))

    async def run_selection_consumer(self) -> None:
        """consumer loop for selection events."""
        logger.info("starting selection consumer loop")

        last_flush = time.time()
        while self._running:
            try:
                async with self._selection_lock:
                    remaining_capacity = self._settings.batch_size - len(self._selection_pending)

                messages = await self._selection_consumer.consume(
                    batch_size=max(1, remaining_capacity),
                    block_ms=100,
                )

                async with self._selection_lock:
                    if messages:
                        self._selection_pending.extend(messages)

                    elapsed = time.time() - last_flush
                    batch_full = len(self._selection_pending) >= self._settings.batch_size
                    time_to_flush = elapsed >= self._settings.flush_interval_sec

                    should_flush = self._selection_pending and (batch_full or time_to_flush)

                    if should_flush:
                        await self._process_selection_batch(self._selection_pending)
                        self._selection_pending = []
                        last_flush = time.time()

            except Exception as e:  # noqa
                logger.error("error processing selection batch: %s", e)
                await asyncio.sleep(1)

    async def run_feedback_consumer(self) -> None:
        """consumer loop for feedback events."""
        logger.info("starting feedback consumer loop")

        last_flush = time.time()
        while self._running:
            try:
                async with self._feedback_lock:
                    remaining_capacity = self._settings.batch_size - len(self._feedback_pending)

                messages = await self._feedback_consumer.consume(
                    batch_size=max(1, remaining_capacity),
                    block_ms=100,
                )

                async with self._feedback_lock:
                    if messages:
                        self._feedback_pending.extend(messages)

                    elapsed = time.time() - last_flush
                    batch_full = len(self._feedback_pending) >= self._settings.batch_size
                    time_to_flush = elapsed >= self._settings.flush_interval_sec

                    should_flush = self._feedback_pending and (batch_full or time_to_flush)

                    if should_flush:
                        await self._process_feedback_batch(self._feedback_pending)
                        self._feedback_pending = []
                        last_flush = time.time()

            except Exception as e:  # noqa
                logger.error("error processing feedback batch: %s", e)
                await asyncio.sleep(1)

    def get_stats(
        self, tenant_id: str | None = None, experiment_id: str | None = None
    ) -> list[dict]:
        """get stats, optionally filtered by tenant_id and/or experiment_id."""
        if tenant_id and experiment_id:
            stats_key = f"{tenant_id}:{experiment_id}"
            stats = self._stats.get(stats_key)
            if stats:
                return [{"tenant_id": tenant_id, "experiment_id": experiment_id, **stats}]
            return []

        results = []
        for key, stats in self._stats.items():
            parts = key.split(":", 1)
            if len(parts) == 2:
                t_id, e_id = parts
                if tenant_id and t_id != tenant_id:
                    continue
                if experiment_id and e_id != experiment_id:
                    continue
                results.append({"tenant_id": t_id, "experiment_id": e_id, **stats})
        return results

    async def health(self) -> bool:
        try:
            return self._clickhouse.health_check()
        except Exception:  # noqa
            return False
