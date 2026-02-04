import pytest
from unittest.mock import AsyncMock
from unittest.mock import Mock

from qbrixstore.redis.streams import FeedbackEvent


@pytest.fixture
def tenant_id():
    """default tenant id for testing"""
    return "tenant-001"


@pytest.fixture
def mock_redis_client():
    """mock redis client for param storage and experiment retrieval."""
    client = AsyncMock()
    client.client = AsyncMock()
    client.client.ping = AsyncMock(return_value=True)
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.get_experiment = AsyncMock()
    client.get_params = AsyncMock()
    client.set_params = AsyncMock()
    return client


@pytest.fixture
def mock_stream_consumer():
    """mock redis stream consumer."""
    consumer = AsyncMock()
    consumer.connect = AsyncMock()
    consumer.close = AsyncMock()
    consumer.consume = AsyncMock(return_value=[])
    consumer.ack = AsyncMock()
    consumer.get_pending_count = AsyncMock(return_value=0)
    consumer.claim_pending = AsyncMock(return_value=[])
    return consumer


@pytest.fixture
def sample_feedback_event(tenant_id):
    """create a sample feedback event."""
    return FeedbackEvent(
        tenant_id=tenant_id,
        experiment_id="exp-001",
        request_id="req-001",
        arm_index=0,
        reward=1.0,
        context_id="ctx-001",
        context_vector=[0.5, 0.3, 0.2],
        context_metadata={"user": "test"},
        timestamp_ms=1234567890,
    )


@pytest.fixture
def sample_experiment_record():
    """create a sample experiment record from redis."""
    return {
        "id": "exp-001",
        "policy": "BetaTSPolicy",
        "policy_params": {},
        "pool": {
            "id": "pool-001",
            "arms": [
                {"id": "arm-0", "index": 0},
                {"id": "arm-1", "index": 1},
                {"id": "arm-2", "index": 2},
            ]
        }
    }


@pytest.fixture
def sample_beta_ts_params():
    """create sample beta thompson sampling params."""
    return {
        "num_arms": 3,
        "alpha": [1.0, 1.0, 1.0],
        "beta": [1.0, 1.0, 1.0],
        "T": [0, 0, 0],
    }