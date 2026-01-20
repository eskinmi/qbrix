import asyncio
from collections import defaultdict

from qbrixstore.redis.client import RedisClient
from qbrixstore.redis.streams import RedisStreamConsumer
from qbrixstore.config import RedisSettings

from cortexsvc.config import CortexSettings
from cortexsvc.trainer import BatchTrainer


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

        self._consumer = RedisStreamConsumer(
            redis_settings, self._settings.consumer_name
        )
        await self._consumer.connect()

        self._trainer = BatchTrainer(self._redis)
        self._running = True

    async def stop(self) -> None:
        self._running = False
        if self._consumer:
            await self._consumer.close()
        if self._redis:
            await self._redis.close()

    async def run_consumer(self) -> None:
        while self._running:
            try:
                messages = await self._consumer.consume(
                    batch_size=self._settings.batch_size,
                    block_ms=self._settings.batch_timeout_ms,
                )

                if not messages:
                    continue

                message_ids = [mid for mid, _ in messages]
                events = [event for _, event in messages]

                ledger = await self._trainer.train_batch(events)

                for experiment_id, count in ledger.items():
                    self._stats[experiment_id]["total"] += count
                    import time

                    self._stats[experiment_id]["last_train"] = int(time.time() * 1000)

                await self._consumer.ack(message_ids)

            except Exception as e:  # noqa
                print(f"Error processing batch: {e}")
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
