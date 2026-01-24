from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from qbrixlog import get_logger
from qbrixstore.redis.client import RedisClient
from qbrixstore.redis.streams import FeedbackEvent
from qbrixstore.redis.streams import RedisStreamConsumer
from qbrixstore.config import RedisSettings

from cortexsvc.config import CortexSettings
from cortexsvc.trainer import BatchTrainer

logger = get_logger(__name__)


class CortexService:
    def __init__(self, settings: CortexSettings):
        self._settings = settings
        self._redis: RedisClient | None = None
        self._consumer: RedisStreamConsumer | None = None
        self._trainer: BatchTrainer | None = None
        self._stats: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "pending": 0, "last_train": 0}
        )
        self._running = False
        self._pending: list[tuple[str, FeedbackEvent]] = []

    async def start(self) -> None:
        redis_settings = RedisSettings(
            host=self._settings.redis_host,
            port=self._settings.redis_port,
            password=self._settings.redis_password,
            db=self._settings.redis_db,
            stream_name=self._settings.stream_name,
            consumer_group=self._settings.consumer_group,
        )
        self._redis = RedisClient(redis_settings)
        await self._redis.connect()
        logger.info("connected to redis at %s:%s", self._settings.redis_host, self._settings.redis_port)

        self._consumer = RedisStreamConsumer(
            redis_settings, self._settings.consumer_name
        )
        await self._consumer.connect()
        logger.info("stream consumer started: %s", self._settings.consumer_name)

        self._trainer = BatchTrainer(self._redis)
        self._running = True

    async def stop(self) -> None:
        self._running = False

        if self._pending:
            logger.info("flushing %d pending messages before shutdown", len(self._pending))
            try:
                await self._process_batch(self._pending)
                self._pending = []
            except Exception as e:  # noqa
                logger.error("failed to flush pending messages: %s", e)

        if self._consumer:
            await self._consumer.close()
        if self._redis:
            await self._redis.close()
        logger.info("cortex service stopped")

    async def _recover_pending(self) -> None:
        """recover and process any pending messages from previous runs."""
        pending_count = await self._consumer.get_pending_count()
        if pending_count == 0:
            return

        logger.info("recovering %d pending messages from previous run", pending_count)

        while True:
            messages = await self._consumer.claim_pending(
                count=self._settings.batch_size,
                min_idle_ms=0,
            )

            if not messages:
                break

            await self._process_batch(messages)

        logger.info("pending message recovery complete")

    async def _process_batch(
        self, messages: list[tuple[str, FeedbackEvent]]
    ) -> None:
        """process a batch of messages: train and ack."""
        message_ids = [mid for mid, _ in messages]
        events = [event for _, event in messages]
        ledger = await self._trainer.train(events)
        for experiment_id, count in ledger.items():
            self._stats[experiment_id]["total"] += count
            self._stats[experiment_id]["last_train"] = int(time.time() * 1000)
        await self._consumer.ack(message_ids)
        logger.info(
            "trained batch: %d events across %d experiments",
            len(events),
            len(ledger),
        )

    async def run_consumer(self) -> None:
        logger.info("starting feedback consumer loop")

        await self._recover_pending()

        last_flush = time.time()
        while self._running:
            try:
                remaining_capacity = self._settings.batch_size - len(self._pending)
                messages = await self._consumer.consume(
                    batch_size=max(1, remaining_capacity),
                    block_ms=100,
                )

                if messages:
                    self._pending.extend(messages)

                elapsed = time.time() - last_flush
                batch_full = len(self._pending) >= self._settings.batch_size
                time_to_flush = elapsed >= self._settings.flush_interval_sec

                if not self._pending or (not batch_full and not time_to_flush):
                    continue

                await self._process_batch(self._pending)

                self._pending = []
                last_flush = time.time()

            except Exception as e:  # noqa
                logger.error("error processing batch: %s", e)
                await asyncio.sleep(1)

    @staticmethod
    async def flush_batch(experiment_id: str | None = None) -> int:  # noqa
        return 0

    def get_stats(self, experiment_id: str | None = None) -> list[dict]:
        if experiment_id:
            stats = self._stats.get(experiment_id)
            if stats:
                return [{"experiment_id": experiment_id, **stats}]
            return []
        return [{"experiment_id": k, **v} for k, v in self._stats.items()]

    async def health(self) -> bool:
        try:
            await self._redis.client.ping()
            return True
        except Exception:  # noqa
            return False