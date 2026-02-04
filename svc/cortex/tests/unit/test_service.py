import asyncio
import pytest
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from cortexsvc.config import CortexSettings
from cortexsvc.service import CortexService
from qbrixstore.redis.streams import FeedbackEvent


class TestCortexServiceInit:
    """test cortex service initialization."""

    def test_init_stores_settings(self):
        # arrange
        settings = CortexSettings(redis_host="test-redis", redis_port=6380)

        # act
        service = CortexService(settings)

        # assert
        assert service._settings is settings
        assert service._redis is None
        assert service._consumer is None
        assert service._trainer is None
        assert service._running is False
        assert service._pending == []

    def test_init_initializes_stats_dict(self):
        # arrange
        settings = CortexSettings()

        # act
        service = CortexService(settings)

        # assert
        assert isinstance(service._stats, dict)
        # accessing a non-existent key should create default stats
        stats = service._stats["test-exp"]
        assert stats["total"] == 0
        assert stats["pending"] == 0
        assert stats["last_train"] == 0


class TestCortexServiceStart:
    """test service startup."""

    @pytest.mark.asyncio
    async def test_start_connects_to_redis(self, mock_redis_client, mock_stream_consumer):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                # act
                await service.start()

                # assert
                mock_redis_client.connect.assert_called_once()
                mock_stream_consumer.connect.assert_called_once()
                assert service._redis is mock_redis_client
                assert service._consumer is mock_stream_consumer
                assert service._running is True

    @pytest.mark.asyncio
    async def test_start_creates_batch_trainer(self, mock_redis_client, mock_stream_consumer):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                # act
                await service.start()

                # assert
                assert service._trainer is not None


class TestCortexServiceStop:
    """test service shutdown."""

    @pytest.mark.asyncio
    async def test_stop_sets_running_to_false(self, mock_redis_client, mock_stream_consumer):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                await service.start()

                # act
                await service.stop()

                # assert
                assert service._running is False

    @pytest.mark.asyncio
    async def test_stop_closes_connections(self, mock_redis_client, mock_stream_consumer):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                await service.start()

                # act
                await service.stop()

                # assert
                mock_consumer = service._consumer
                mock_redis = service._redis
                mock_consumer.close.assert_called_once()
                mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_flushes_pending_messages(self, mock_redis_client, mock_stream_consumer, sample_feedback_event):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                with patch("cortexsvc.service.BatchTrainer") as mock_trainer_cls:
                    mock_trainer = AsyncMock()
                    mock_trainer.train = AsyncMock(return_value={("tenant-001", "exp-001"): 1})
                    mock_trainer_cls.return_value = mock_trainer

                    await service.start()
                    service._pending = [("msg-1", sample_feedback_event)]

                    # act
                    await service.stop()

                    # assert
                    mock_trainer.train.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_handles_flush_errors_gracefully(self, mock_redis_client, mock_stream_consumer, sample_feedback_event):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                with patch("cortexsvc.service.BatchTrainer") as mock_trainer_cls:
                    mock_trainer = AsyncMock()
                    mock_trainer.train = AsyncMock(side_effect=Exception("flush error"))
                    mock_trainer_cls.return_value = mock_trainer

                    await service.start()
                    service._pending = [("msg-1", sample_feedback_event)]

                    # act - should not raise
                    await service.stop()

                    # assert
                    assert service._running is False


class TestCortexServiceRecoverPending:
    """test recovery of pending messages from previous runs."""

    @pytest.mark.asyncio
    async def test_recover_pending_with_no_pending_messages(self, mock_redis_client, mock_stream_consumer):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                await service.start()
                mock_stream_consumer.get_pending_count.return_value = 0

                # act
                await service._recover_pending()

                # assert
                mock_stream_consumer.claim_pending.assert_not_called()

    @pytest.mark.asyncio
    async def test_recover_pending_claims_and_processes_messages(
        self, mock_redis_client, mock_stream_consumer, sample_feedback_event
    ):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                with patch("cortexsvc.service.BatchTrainer") as mock_trainer_cls:
                    mock_trainer = AsyncMock()
                    mock_trainer.train = AsyncMock(return_value={("tenant-001", "exp-001"): 2})
                    mock_trainer_cls.return_value = mock_trainer

                    await service.start()

                    mock_stream_consumer.get_pending_count.return_value = 2
                    mock_stream_consumer.claim_pending.side_effect = [
                        [("msg-1", sample_feedback_event), ("msg-2", sample_feedback_event)],
                        []
                    ]

                    # act
                    await service._recover_pending()

                    # assert
                    assert mock_stream_consumer.claim_pending.call_count == 2
                    mock_trainer.train.assert_called_once()


class TestCortexServiceProcessBatch:
    """test batch processing logic."""

    @pytest.mark.asyncio
    async def test_process_batch_trains_and_acks(
        self, mock_redis_client, mock_stream_consumer, sample_feedback_event
    ):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                with patch("cortexsvc.service.BatchTrainer") as mock_trainer_cls:
                    mock_trainer = AsyncMock()
                    mock_trainer.train = AsyncMock(return_value={("tenant-001", "exp-001"): 1})
                    mock_trainer_cls.return_value = mock_trainer

                    await service.start()

                    messages = [("msg-1", sample_feedback_event)]

                    # act
                    await service._process_batch(messages)

                    # assert
                    mock_trainer.train.assert_called_once()
                    mock_stream_consumer.ack.assert_called_once_with(["msg-1"])

    @pytest.mark.asyncio
    async def test_process_batch_updates_stats(
        self, mock_redis_client, mock_stream_consumer, sample_feedback_event
    ):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                with patch("cortexsvc.service.BatchTrainer") as mock_trainer_cls:
                    mock_trainer = AsyncMock()
                    mock_trainer.train = AsyncMock(return_value={("tenant-001", "exp-001"): 2})
                    mock_trainer_cls.return_value = mock_trainer

                    await service.start()

                    messages = [
                        ("msg-1", sample_feedback_event),
                        ("msg-2", sample_feedback_event)
                    ]

                    # act
                    await service._process_batch(messages)

                    # assert
                    assert service._stats["tenant-001:exp-001"]["total"] == 2
                    assert service._stats["tenant-001:exp-001"]["last_train"] > 0


class TestCortexServiceRunConsumer:
    """test consumer loop."""

    @pytest.mark.asyncio
    async def test_run_consumer_recovers_pending_on_start(
        self, mock_redis_client, mock_stream_consumer
    ):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                await service.start()

                mock_stream_consumer.get_pending_count.return_value = 0

                # consume must yield control to event loop for stop task to run
                call_count = 0
                async def consume_side_effect(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    await asyncio.sleep(0.01)
                    if call_count >= 1:
                        service._running = False
                    return []

                mock_stream_consumer.consume.side_effect = consume_side_effect

                # act
                await service.run_consumer()

                # assert
                mock_stream_consumer.get_pending_count.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_consumer_processes_incoming_messages(
        self, mock_redis_client, mock_stream_consumer, sample_feedback_event
    ):
        # arrange
        settings = CortexSettings(batch_size=2)
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                with patch("cortexsvc.service.BatchTrainer") as mock_trainer_cls:
                    mock_trainer = AsyncMock()
                    mock_trainer.train = AsyncMock(return_value={("tenant-001", "exp-001"): 2})
                    mock_trainer_cls.return_value = mock_trainer

                    await service.start()

                    mock_stream_consumer.get_pending_count.return_value = 0

                    call_count = 0
                    def consume_side_effect(*args, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        if call_count == 1:
                            return [("msg-1", sample_feedback_event)]
                        elif call_count == 2:
                            return [("msg-2", sample_feedback_event)]
                        else:
                            service._running = False
                            return []

                    mock_stream_consumer.consume.side_effect = consume_side_effect

                    # act
                    await service.run_consumer()

                    # assert
                    # batch should be flushed when it reaches batch_size
                    mock_trainer.train.assert_called()

    @pytest.mark.asyncio
    async def test_run_consumer_flushes_on_timeout(
        self, mock_redis_client, mock_stream_consumer, sample_feedback_event
    ):
        # arrange
        settings = CortexSettings(batch_size=10, flush_interval_sec=5)
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                with patch("cortexsvc.service.BatchTrainer") as mock_trainer_cls:
                    mock_trainer = AsyncMock()
                    mock_trainer.train = AsyncMock(return_value={("tenant-001", "exp-001"): 1})
                    mock_trainer_cls.return_value = mock_trainer

                    await service.start()

                    mock_stream_consumer.get_pending_count.return_value = 0

                    # mock time to simulate timeout without actual waiting
                    fake_time = [100.0]

                    def get_fake_time():
                        return fake_time[0]

                    call_count = 0
                    async def consume_side_effect(*args, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        await asyncio.sleep(0.001)
                        if call_count == 1:
                            return [("msg-1", sample_feedback_event)]
                        elif call_count == 2:
                            # advance time past flush interval
                            fake_time[0] += 10
                            return []
                        else:
                            service._running = False
                            return []

                    mock_stream_consumer.consume.side_effect = consume_side_effect

                    # act
                    with patch("cortexsvc.service.time.time", side_effect=get_fake_time):
                        await service.run_consumer()

                    # assert
                    # should flush due to timeout even though batch is not full
                    mock_trainer.train.assert_called()

    @pytest.mark.asyncio
    async def test_run_consumer_handles_errors_gracefully(
        self, mock_redis_client, mock_stream_consumer
    ):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                await service.start()

                mock_stream_consumer.get_pending_count.return_value = 0

                call_count = 0
                async def consume_side_effect(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        raise Exception("consume error")
                    else:
                        service._running = False
                        return []

                mock_stream_consumer.consume.side_effect = consume_side_effect

                # act - should not raise
                await service.run_consumer()

                # assert
                assert service._running is False


class TestCortexServiceFlushBatch:
    """test manual batch flushing."""

    @pytest.mark.asyncio
    async def test_flush_batch_with_empty_pending_returns_zero(
        self, mock_redis_client, mock_stream_consumer
    ):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                await service.start()

                # act
                count = await service.flush_batch()

                # assert
                assert count == 0

    @pytest.mark.asyncio
    async def test_flush_batch_without_experiment_id_flushes_all(
        self, mock_redis_client, mock_stream_consumer, sample_feedback_event
    ):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                with patch("cortexsvc.service.BatchTrainer") as mock_trainer_cls:
                    mock_trainer = AsyncMock()
                    mock_trainer.train = AsyncMock(return_value={("tenant-001", "exp-001"): 2})
                    mock_trainer_cls.return_value = mock_trainer

                    await service.start()

                    service._pending = [
                        ("msg-1", sample_feedback_event),
                        ("msg-2", sample_feedback_event)
                    ]

                    # act
                    count = await service.flush_batch()

                    # assert
                    assert count == 2
                    assert len(service._pending) == 0
                    mock_trainer.train.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_batch_with_experiment_id_filters_events(
        self, mock_redis_client, mock_stream_consumer
    ):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                with patch("cortexsvc.service.BatchTrainer") as mock_trainer_cls:
                    mock_trainer = AsyncMock()
                    mock_trainer.train = AsyncMock(return_value={("tenant-001", "exp-001"): 1})
                    mock_trainer_cls.return_value = mock_trainer

                    await service.start()

                    event1 = FeedbackEvent(
                        tenant_id="tenant-001",
                        experiment_id="exp-001",
                        request_id="req-001",
                        arm_index=0,
                        reward=1.0,
                        context_id="ctx-001",
                        context_vector=[0.5],
                        context_metadata={},
                        timestamp_ms=1234567890,
                    )
                    event2 = FeedbackEvent(
                        tenant_id="tenant-001",
                        experiment_id="exp-002",
                        request_id="req-002",
                        arm_index=1,
                        reward=0.5,
                        context_id="ctx-002",
                        context_vector=[0.3],
                        context_metadata={},
                        timestamp_ms=1234567891,
                    )

                    service._pending = [
                        ("msg-1", event1),
                        ("msg-2", event2)
                    ]

                    # act
                    count = await service.flush_batch(experiment_id="exp-001")

                    # assert
                    assert count == 1
                    assert len(service._pending) == 1
                    assert service._pending[0][1].experiment_id == "exp-002"


class TestCortexServiceGetStats:
    """test stats retrieval."""

    def test_get_stats_without_experiment_id_returns_all(self, tenant_id):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)
        service._stats[f"{tenant_id}:exp-001"] = {"total": 10, "pending": 0, "last_train": 123}
        service._stats[f"{tenant_id}:exp-002"] = {"total": 5, "pending": 2, "last_train": 456}

        # act
        stats = service.get_stats()

        # assert
        assert len(stats) == 2
        exp_ids = {s["experiment_id"] for s in stats}
        assert "exp-001" in exp_ids
        assert "exp-002" in exp_ids

    def test_get_stats_with_experiment_id_returns_single(self, tenant_id):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)
        service._stats[f"{tenant_id}:exp-001"] = {"total": 10, "pending": 0, "last_train": 123}

        # act
        stats = service.get_stats(tenant_id=tenant_id, experiment_id="exp-001")

        # assert
        assert len(stats) == 1
        assert stats[0]["experiment_id"] == "exp-001"
        assert stats[0]["total"] == 10

    def test_get_stats_with_missing_experiment_id_returns_empty(self, tenant_id):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        # act
        stats = service.get_stats(tenant_id=tenant_id, experiment_id="exp-999")

        # assert
        assert stats == []


class TestCortexServiceHealth:
    """test health check."""

    @pytest.mark.asyncio
    async def test_health_returns_true_when_redis_is_reachable(
        self, mock_redis_client, mock_stream_consumer
    ):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                await service.start()
                mock_redis_client.client.ping.return_value = True

                # act
                healthy = await service.health()

                # assert
                assert healthy is True

    @pytest.mark.asyncio
    async def test_health_returns_false_when_redis_is_unreachable(
        self, mock_redis_client, mock_stream_consumer
    ):
        # arrange
        settings = CortexSettings()
        service = CortexService(settings)

        with patch("cortexsvc.service.RedisClient", return_value=mock_redis_client):
            with patch("cortexsvc.service.RedisStreamConsumer", return_value=mock_stream_consumer):
                await service.start()
                mock_redis_client.client.ping.side_effect = Exception("connection error")

                # act
                healthy = await service.health()

                # assert
                assert healthy is False
