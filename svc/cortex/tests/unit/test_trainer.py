import pytest
from unittest.mock import AsyncMock

from cortexsvc.trainer import BatchTrainer
from cortexsvc.trainer import PROTOCOL_MAP
from cortexsvc.trainer import _build_protocol_map
from qbrixstore.redis.streams import FeedbackEvent


class TestProtocolMapBuilding:
    """test protocol registry construction."""

    def test_build_protocol_map_returns_dict(self):
        # arrange & act
        protocol_map = _build_protocol_map()

        # assert
        assert isinstance(protocol_map, dict)
        assert len(protocol_map) > 0

    def test_protocol_map_contains_beta_ts(self):
        # arrange & act
        # assert
        assert "BetaTSProtocol" in PROTOCOL_MAP
        assert PROTOCOL_MAP["BetaTSProtocol"].name == "BetaTSProtocol"

    def test_protocol_map_contains_ucb1_tuned(self):
        # arrange & act
        # assert
        assert "UCB1TunedProtocol" in PROTOCOL_MAP
        assert PROTOCOL_MAP["UCB1TunedProtocol"].name == "UCB1TunedProtocol"


class TestBatchTrainerInit:
    """test batch trainer initialization."""

    def test_init_stores_redis_client(self, mock_redis_client):
        # arrange & act
        trainer = BatchTrainer(mock_redis_client)

        # assert
        assert trainer._redis is mock_redis_client


class TestBatchTrainerTrain:
    """test batch training orchestration."""

    @pytest.mark.asyncio
    async def test_train_with_empty_events_returns_empty_ledger(
        self, mock_redis_client
    ):
        # arrange
        trainer = BatchTrainer(mock_redis_client)
        events = []

        # act
        ledger = await trainer.train(events)

        # assert
        assert ledger == {}
        mock_redis_client.get_experiment.assert_not_called()

    @pytest.mark.asyncio
    async def test_train_with_single_event_processes_one_experiment(
        self,
        tenant_id,
        mock_redis_client,
        sample_feedback_event,
        sample_experiment_record,
        sample_beta_ts_params,
    ):
        # arrange
        trainer = BatchTrainer(mock_redis_client)
        mock_redis_client.get_experiment.return_value = sample_experiment_record
        mock_redis_client.get_params.return_value = sample_beta_ts_params

        # act
        ledger = await trainer.train([sample_feedback_event])

        # assert
        assert (tenant_id, "exp-001") in ledger
        assert ledger[(tenant_id, "exp-001")] == 1
        mock_redis_client.get_experiment.assert_called_once_with(tenant_id, "exp-001")
        mock_redis_client.set_params.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_with_multiple_events_same_experiment(
        self,
        tenant_id,
        mock_redis_client,
        sample_feedback_event,
        sample_experiment_record,
        sample_beta_ts_params,
    ):
        # arrange
        trainer = BatchTrainer(mock_redis_client)
        mock_redis_client.get_experiment.return_value = sample_experiment_record
        mock_redis_client.get_params.return_value = sample_beta_ts_params

        events = [
            sample_feedback_event,
            FeedbackEvent(
                tenant_id=tenant_id,
                experiment_id="exp-001",
                request_id="req-002",
                arm_index=1,
                reward=0.0,
                context_id="ctx-002",
                context_vector=[0.1, 0.2, 0.3],
                context_metadata={},
                timestamp_ms=1234567891,
            ),
        ]

        # act
        ledger = await trainer.train(events)

        # assert
        assert ledger[(tenant_id, "exp-001")] == 2
        mock_redis_client.get_experiment.assert_called_once()
        mock_redis_client.set_params.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_with_events_from_different_experiments(
        self, tenant_id, mock_redis_client, sample_experiment_record, sample_beta_ts_params
    ):
        # arrange
        trainer = BatchTrainer(mock_redis_client)
        mock_redis_client.get_experiment.return_value = sample_experiment_record
        mock_redis_client.get_params.return_value = sample_beta_ts_params

        events = [
            FeedbackEvent(
                tenant_id=tenant_id,
                experiment_id="exp-001",
                request_id="req-001",
                arm_index=0,
                reward=1.0,
                context_id="ctx-001",
                context_vector=[0.5, 0.3, 0.2],
                context_metadata={},
                timestamp_ms=1234567890,
            ),
            FeedbackEvent(
                tenant_id=tenant_id,
                experiment_id="exp-002",
                request_id="req-002",
                arm_index=1,
                reward=0.5,
                context_id="ctx-002",
                context_vector=[0.1, 0.2, 0.3],
                context_metadata={},
                timestamp_ms=1234567891,
            ),
        ]

        # act
        ledger = await trainer.train(events)

        # assert
        assert (tenant_id, "exp-001") in ledger
        assert (tenant_id, "exp-002") in ledger
        assert ledger[(tenant_id, "exp-001")] == 1
        assert ledger[(tenant_id, "exp-002")] == 1
        assert mock_redis_client.get_experiment.call_count == 2
        assert mock_redis_client.set_params.call_count == 2


class TestBatchTrainerTrainExperiment:
    """test single experiment training logic."""

    @pytest.mark.asyncio
    async def test_train_experiment_with_missing_experiment_returns_zero(
        self, tenant_id, mock_redis_client, sample_feedback_event
    ):
        # arrange
        trainer = BatchTrainer(mock_redis_client)
        mock_redis_client.get_experiment.return_value = None

        # act
        count = await trainer._train_experiment(tenant_id, "exp-001", [sample_feedback_event])

        # assert
        assert count == 0
        mock_redis_client.get_params.assert_not_called()
        mock_redis_client.set_params.assert_not_called()

    @pytest.mark.asyncio
    async def test_train_experiment_with_unknown_protocol_returns_zero(
        self, tenant_id, mock_redis_client, sample_feedback_event
    ):
        # arrange
        trainer = BatchTrainer(mock_redis_client)
        mock_redis_client.get_experiment.return_value = {
            "id": "exp-001",
            "protocol": "unknown_protocol",
            "pool": {"arms": [{"id": "arm-0", "index": 0}]}
        }

        # act
        count = await trainer._train_experiment(tenant_id, "exp-001", [sample_feedback_event])

        # assert
        assert count == 0
        mock_redis_client.set_params.assert_not_called()

    @pytest.mark.asyncio
    async def test_train_experiment_initializes_params_when_missing(
        self, tenant_id, mock_redis_client, sample_feedback_event, sample_experiment_record
    ):
        # arrange
        trainer = BatchTrainer(mock_redis_client)
        mock_redis_client.get_experiment.return_value = sample_experiment_record
        mock_redis_client.get_params.return_value = None

        # act
        count = await trainer._train_experiment(tenant_id, "exp-001", [sample_feedback_event])

        # assert
        assert count == 1
        mock_redis_client.set_params.assert_called_once()

        # verify params were initialized
        set_params_call = mock_redis_client.set_params.call_args
        params = set_params_call[0][2]  # tenant_id, experiment_id, params
        assert "alpha" in params
        assert "beta" in params
        assert len(params["alpha"]) == 3

    @pytest.mark.asyncio
    async def test_train_experiment_uses_existing_params(
        self,
        tenant_id,
        mock_redis_client,
        sample_feedback_event,
        sample_experiment_record,
        sample_beta_ts_params,
    ):
        # arrange
        trainer = BatchTrainer(mock_redis_client)
        mock_redis_client.get_experiment.return_value = sample_experiment_record
        mock_redis_client.get_params.return_value = sample_beta_ts_params

        # act
        count = await trainer._train_experiment(tenant_id, "exp-001", [sample_feedback_event])

        # assert
        assert count == 1
        mock_redis_client.get_params.assert_called_once_with(tenant_id, "exp-001")
        mock_redis_client.set_params.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_experiment_updates_params_correctly(
        self,
        tenant_id,
        mock_redis_client,
        sample_feedback_event,
        sample_experiment_record,
        sample_beta_ts_params,
    ):
        # arrange
        trainer = BatchTrainer(mock_redis_client)
        mock_redis_client.get_experiment.return_value = sample_experiment_record
        mock_redis_client.get_params.return_value = sample_beta_ts_params

        # act
        await trainer._train_experiment(tenant_id, "exp-001", [sample_feedback_event])

        # assert
        set_params_call = mock_redis_client.set_params.call_args
        updated_params = set_params_call[0][2]  # tenant_id, experiment_id, params

        # for beta_ts with reward=1.0 on arm 0, alpha[0] should increase
        assert updated_params["alpha"][0] > sample_beta_ts_params["alpha"][0]

    @pytest.mark.asyncio
    async def test_train_experiment_processes_multiple_events(
        self,
        tenant_id,
        mock_redis_client,
        sample_experiment_record,
        sample_beta_ts_params,
    ):
        # arrange
        trainer = BatchTrainer(mock_redis_client)
        mock_redis_client.get_experiment.return_value = sample_experiment_record
        mock_redis_client.get_params.return_value = sample_beta_ts_params

        events = [
            FeedbackEvent(
                tenant_id=tenant_id,
                experiment_id="exp-001",
                request_id="req-001",
                arm_index=0,
                reward=1.0,
                context_id="ctx-001",
                context_vector=[0.5, 0.3, 0.2],
                context_metadata={},
                timestamp_ms=1234567890,
            ),
            FeedbackEvent(
                tenant_id=tenant_id,
                experiment_id="exp-001",
                request_id="req-002",
                arm_index=1,
                reward=0.0,
                context_id="ctx-002",
                context_vector=[0.1, 0.2, 0.3],
                context_metadata={},
                timestamp_ms=1234567891,
            ),
            FeedbackEvent(
                tenant_id=tenant_id,
                experiment_id="exp-001",
                request_id="req-003",
                arm_index=0,
                reward=1.0,
                context_id="ctx-003",
                context_vector=[0.2, 0.4, 0.4],
                context_metadata={},
                timestamp_ms=1234567892,
            ),
        ]

        # act
        count = await trainer._train_experiment(tenant_id, "exp-001", events)

        # assert
        assert count == 3
        mock_redis_client.set_params.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_experiment_with_protocol_params(
        self, tenant_id, mock_redis_client, sample_feedback_event
    ):
        # arrange
        trainer = BatchTrainer(mock_redis_client)
        experiment_record = {
            "id": "exp-001",
            "protocol": "BetaTSProtocol",
            "protocol_params": {"alpha_prior": 2.0, "beta_prior": 2.0},
            "pool": {
                "arms": [
                    {"id": "arm-0", "index": 0},
                    {"id": "arm-1", "index": 1},
                ]
            }
        }
        mock_redis_client.get_experiment.return_value = experiment_record
        mock_redis_client.get_params.return_value = None

        # act
        count = await trainer._train_experiment(tenant_id, "exp-001", [sample_feedback_event])

        # assert
        assert count == 1

        # verify protocol params were passed to init_params
        set_params_call = mock_redis_client.set_params.call_args
        params = set_params_call[0][2]  # tenant_id, experiment_id, params
        # with alpha_prior=2.0, beta_prior=2.0, and reward=1.0 on arm 0
        # alpha[0] should be 2.0 + 1 = 3.0, beta[0] stays at 2.0
        assert params["alpha"][0] == 3.0
        assert params["beta"][0] == 2.0
