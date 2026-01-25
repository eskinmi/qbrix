"""Unit tests for Agent class."""

import pytest
from unittest.mock import Mock

from qbrixcore.agent import Agent
from qbrixcore.param.backend import InMemoryParamBackend
from qbrixcore.protoc.stochastic.ts import BetaTSProtocol
from qbrixcore.protoc.stochastic.ucb import UCB1TunedProtocol
from qbrixcore.callback import BaseCallback


class TestAgent:
    def test_agent_creation_minimal(self, pool_with_three_arms):
        """test agent creation with minimal required fields."""
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol
        )

        assert agent.experiment_id == "exp-123"
        assert agent.pool == pool_with_three_arms
        assert agent.protocol == BetaTSProtocol
        assert agent.init_params == {}
        assert isinstance(agent.param_backend, InMemoryParamBackend)
        assert agent.id is not None
        assert agent.callbacks == []

    def test_agent_creation_with_custom_backend(self, pool_with_three_arms):
        """test agent creation with custom parameter backend."""
        backend = InMemoryParamBackend()
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol,
            param_backend=backend
        )

        assert agent.param_backend is backend

    def test_agent_creation_with_init_params(self, pool_with_three_arms):
        """test agent creation with protocol initialization parameters."""
        init_params = {"alpha_prior": 2.0, "beta_prior": 3.0}
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol,
            init_params=init_params
        )

        assert agent.init_params == init_params

    def test_agent_unique_ids(self, pool_with_three_arms):
        """test different agents get unique ids."""
        agent1 = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol
        )
        agent2 = Agent(
            experiment_id="exp-456",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol
        )

        assert agent1.id != agent2.id

    def test_agent_select_raises_without_initialized_params(
        self, pool_with_three_arms, context_without_vector
    ):
        """test select raises error when params not initialized."""
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol
        )

        with pytest.raises(RuntimeError, match="param state not found"):
            agent.select(context_without_vector)

    def test_agent_select_returns_valid_arm_index(
        self, pool_with_three_arms, context_without_vector, in_memory_backend
    ):
        """test select returns valid arm index."""
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol,
            param_backend=in_memory_backend
        )

        # initialize params
        params = BetaTSProtocol.init_params(num_arms=3)
        in_memory_backend.set("exp-123", params)

        choice = agent.select(context_without_vector)

        assert isinstance(choice, int)
        assert 0 <= choice < 3

    def test_agent_train_raises_without_initialized_params(
        self, pool_with_three_arms, context_without_vector
    ):
        """test train raises error when params not initialized."""
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol
        )

        with pytest.raises(RuntimeError, match="param state not found"):
            agent.train(context_without_vector, choice=0, reward=1.0)

    def test_agent_train_updates_params(
        self, pool_with_three_arms, context_without_vector, in_memory_backend
    ):
        """test train updates parameters in backend."""
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol,
            param_backend=in_memory_backend
        )

        # initialize params
        params = BetaTSProtocol.init_params(num_arms=3)
        in_memory_backend.set("exp-123", params)

        # train
        updated_params = agent.train(context_without_vector, choice=1, reward=1.0)

        # verify params updated in backend
        stored_params = in_memory_backend.get("exp-123")
        assert stored_params.alpha[1] == 2.0  # success increases alpha
        assert updated_params == stored_params

    def test_agent_select_and_train_workflow(
        self, pool_with_three_arms, context_without_vector, in_memory_backend
    ):
        """test typical select-train workflow."""
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol,
            param_backend=in_memory_backend
        )

        # initialize params
        params = BetaTSProtocol.init_params(num_arms=3)
        in_memory_backend.set("exp-123", params)

        # select
        choice = agent.select(context_without_vector)

        # train on selection
        agent.train(context_without_vector, choice=choice, reward=1.0)

        # select again
        choice2 = agent.select(context_without_vector)
        assert 0 <= choice2 < 3

    def test_agent_with_different_protocol(
        self, pool_with_three_arms, context_without_vector, in_memory_backend
    ):
        """test agent works with different protocol."""
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=UCB1TunedProtocol,
            param_backend=in_memory_backend
        )

        # initialize params
        params = UCB1TunedProtocol.init_params(num_arms=3)
        in_memory_backend.set("exp-123", params)

        choice = agent.select(context_without_vector)
        assert 0 <= choice < 3

        agent.train(context_without_vector, choice=choice, reward=0.5)
        stored_params = in_memory_backend.get("exp-123")
        assert stored_params.round == 1

    def test_agent_multiple_experiments_isolated(self, pool_with_three_arms, context_without_vector):
        """test multiple agents with different experiment ids are isolated."""
        backend = InMemoryParamBackend()

        agent1 = Agent(
            experiment_id="exp-1",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol,
            param_backend=backend
        )
        agent2 = Agent(  # noqa
            experiment_id="exp-2",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol,
            param_backend=backend
        )

        # initialize params for both
        backend.set("exp-1", BetaTSProtocol.init_params(num_arms=3))
        backend.set("exp-2", BetaTSProtocol.init_params(num_arms=3))

        # train agent1
        agent1.train(context_without_vector, choice=0, reward=1.0)

        # agent2 params should be unchanged
        params2 = backend.get("exp-2")
        assert params2.alpha[0] == 1.0  # unchanged

    def test_agent_add_callback_valid(self, pool_with_three_arms):
        """test adding valid callback to agent."""
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol
        )

        mock_callback = Mock(spec=BaseCallback)
        mock_callback.scope = "test"

        agent.add_callback(mock_callback)

        assert len(agent.callbacks) == 1
        assert agent.callbacks[0] == mock_callback

    def test_agent_add_callback_invalid_type(self, pool_with_three_arms):
        """test adding invalid callback raises TypeError."""
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol
        )

        with pytest.raises(TypeError, match="Callback must be an instance"):
            agent.add_callback("not a callback")  # noqa

    def test_agent_add_multiple_callbacks(self, pool_with_three_arms):
        """test adding multiple callbacks to agent."""
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol
        )

        callback1 = Mock(spec=BaseCallback)
        callback1.scope = "test1"
        callback2 = Mock(spec=BaseCallback)
        callback2.scope = "test2"

        agent.add_callback(callback1)
        agent.add_callback(callback2)

        assert len(agent.callbacks) == 2
        assert callback1 in agent.callbacks
        assert callback2 in agent.callbacks

    def test_agent_callbacks_invoked_on_select(
        self, pool_with_three_arms, context_without_vector, in_memory_backend
    ):
        """test callbacks are invoked during select."""
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol,
            param_backend=in_memory_backend
        )

        # initialize params
        in_memory_backend.set("exp-123", BetaTSProtocol.init_params(num_arms=3))

        # add callback
        mock_callback = Mock(spec=BaseCallback)
        mock_callback.scope = "test"
        agent.add_callback(mock_callback)

        # select
        agent.select(context_without_vector)

        # verify callbacks invoked
        mock_callback.on_select_start.assert_called_once()
        mock_callback.on_select_end.assert_called_once()

    def test_agent_callbacks_invoked_on_train(
        self, pool_with_three_arms, context_without_vector, in_memory_backend
    ):
        """test callbacks are invoked during train."""
        agent = Agent(
            experiment_id="exp-123",
            pool=pool_with_three_arms,
            protocol=BetaTSProtocol,
            param_backend=in_memory_backend
        )

        # initialize params
        in_memory_backend.set("exp-123", BetaTSProtocol.init_params(num_arms=3))

        # add callback with tracking
        class TrackingCallback(BaseCallback):
            def __init__(self):
                self.train_start_called = False
                self.train_end_called = False

            @property
            def scope(self):
                return "test"

            def on_train_start(self, agent):  # noqa
                self.train_start_called = True

            def on_train_end(self, agent):  # noqa
                self.train_end_called = True

        callback = TrackingCallback()
        agent.add_callback(callback)

        # train
        agent.train(context_without_vector, choice=0, reward=1.0)

        # verify callbacks invoked
        assert callback.train_start_called is True
        assert callback.train_end_called is True

    def test_agent_with_empty_pool(self, empty_pool, context_without_vector, in_memory_backend):
        """test agent with empty pool can be created but fails on select."""
        agent = Agent(  # noqa
            experiment_id="exp-123",
            pool=empty_pool,
            protocol=BetaTSProtocol,
            param_backend=in_memory_backend
        )

        # initialize params with 0 arms would fail validation
        with pytest.raises(ValueError):
            BetaTSProtocol.init_params(num_arms=0)
