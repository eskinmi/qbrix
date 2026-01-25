"""Unit tests for UCB1TunedProtocol."""

import numpy as np
import pytest

from qbrixcore.protoc.stochastic.ucb import UCB1TunedProtocol
from qbrixcore.protoc.stochastic.ucb import UCB1TunedParamState
from qbrixcore.context import Context


class TestUCB1TunedParamState:
    def test_param_state_creation_minimal(self):
        """test param state creation with minimal args."""
        ps = UCB1TunedParamState(num_arms=3)

        assert ps.num_arms == 3
        assert ps.alpha == 2.0
        assert len(ps.mu) == 3
        assert len(ps.T) == 3
        assert len(ps.rsq) == 3
        assert np.all(ps.mu == 0.0)
        assert np.all(ps.T == 0)
        assert np.all(ps.rsq == 0.0)
        assert ps.round == 0

    def test_param_state_creation_with_custom_alpha(self):
        """test param state creation with custom alpha."""
        ps = UCB1TunedParamState(num_arms=3, alpha=3.0)

        assert ps.alpha == 3.0

    def test_param_state_creation_with_custom_arrays(self):
        """test param state creation with pre-initialized arrays."""
        mu = np.array([0.5, 0.6, 0.7])
        T = np.array([10, 20, 30])
        rsq = np.array([0.3, 0.4, 0.5])

        ps = UCB1TunedParamState(num_arms=3, mu=mu, T=T, rsq=rsq, round=50)

        assert np.array_equal(ps.mu, mu)
        assert np.array_equal(ps.T, T)
        assert np.array_equal(ps.rsq, rsq)
        assert ps.round == 50

    def test_param_state_validation_num_arms_positive(self):
        """test param state validation requires positive num_arms."""
        with pytest.raises(ValueError):
            UCB1TunedParamState(num_arms=0)

    def test_param_state_validation_alpha_positive(self):
        """test param state validation requires positive alpha."""
        with pytest.raises(ValueError):
            UCB1TunedParamState(num_arms=3, alpha=0.0)

    def test_param_state_validation_round_non_negative(self):
        """test param state validation requires non-negative round."""
        with pytest.raises(ValueError):
            UCB1TunedParamState(num_arms=3, round=-1)


class TestUCB1TunedProtocol:
    def test_protocol_name(self):
        """test protocol has correct name."""
        assert UCB1TunedProtocol.name == "UCB1TunedProtocol"

    def test_protocol_param_state_cls(self):
        """test protocol has correct param state class."""
        assert UCB1TunedProtocol.param_state_cls == UCB1TunedParamState

    def test_init_params(self):
        """test init_params creates correct param state."""
        params = UCB1TunedProtocol.init_params(num_arms=4)

        assert isinstance(params, UCB1TunedParamState)
        assert params.num_arms == 4

    def test_select_returns_valid_arm_index(self):
        """test select returns valid arm index."""
        ps = UCB1TunedParamState(num_arms=5)
        ctx = Context()

        arm_index = UCB1TunedProtocol.select(ps, ctx)

        assert isinstance(arm_index, int)
        assert 0 <= arm_index < 5

    def test_select_initial_round_explores_first_unplayed_arm(self):
        """test select prioritizes unplayed arms with infinite upper bound."""
        ps = UCB1TunedParamState(
            num_arms=3,
            mu=np.array([0.5, 0.0, 0.0]),
            T=np.array([10, 0, 0]),
            round=10
        )
        ctx = Context()

        arm_index = UCB1TunedProtocol.select(ps, ctx)

        # should select an unplayed arm (1 or 2) because they have inf upper bound
        assert arm_index in [1, 2]

    def test_select_prefers_higher_mean_arm(self):
        """test select prefers arm with higher mean when sufficient data."""
        ps = UCB1TunedParamState(
            num_arms=3,
            mu=np.array([0.9, 0.1, 0.1]),
            T=np.array([100, 100, 100]),
            rsq=np.array([90.0, 10.0, 10.0]),
            round=300
        )
        ctx = Context()

        # with enough data, arm 0 should be selected
        arm_index = UCB1TunedProtocol.select(ps, ctx)

        assert arm_index == 0

    def test_train_updates_statistics(self):
        """test train updates mean, T, and rsq."""
        ps = UCB1TunedParamState(num_arms=3)
        ctx = Context()

        updated = UCB1TunedProtocol.train(ps, ctx, choice=1, reward=0.8)

        assert updated.T[1] == 1
        assert updated.mu[1] == pytest.approx(0.8)
        assert updated.rsq[1] == pytest.approx(0.64)  # 0.8^2
        assert updated.round == 1
        # other arms unchanged
        assert updated.T[0] == 0
        assert updated.T[2] == 0

    def test_train_incremental_mean_calculation(self):
        """test train uses incremental mean update."""
        ps = UCB1TunedParamState(num_arms=3)
        ctx = Context()

        # first update
        ps = UCB1TunedProtocol.train(ps, ctx, choice=0, reward=1.0)
        assert ps.mu[0] == pytest.approx(1.0)

        # second update
        ps = UCB1TunedProtocol.train(ps, ctx, choice=0, reward=0.5)
        assert ps.mu[0] == pytest.approx(0.75)  # (1.0 + 0.5) / 2

        # third update
        ps = UCB1TunedProtocol.train(ps, ctx, choice=0, reward=0.0)
        assert ps.mu[0] == pytest.approx(0.5)  # (1.0 + 0.5 + 0.0) / 3

    def test_train_increments_round(self):
        """test train increments round counter."""
        ps = UCB1TunedParamState(num_arms=3, round=5)
        ctx = Context()

        updated = UCB1TunedProtocol.train(ps, ctx, choice=0, reward=0.5)

        assert updated.round == 6

    def test_train_does_not_mutate_original_params(self):
        """test train returns new params without mutating original."""
        ps = UCB1TunedParamState(num_arms=3)
        ctx = Context()
        original_mu = ps.mu.copy()
        original_T = ps.T.copy()
        original_rsq = ps.rsq.copy()
        original_round = ps.round

        updated = UCB1TunedProtocol.train(ps, ctx, choice=1, reward=0.8)

        # original unchanged
        assert np.array_equal(ps.mu, original_mu)
        assert np.array_equal(ps.T, original_T)
        assert np.array_equal(ps.rsq, original_rsq)
        assert ps.round == original_round
        # updated changed
        assert updated.mu[1] == 0.8

    def test_train_multiple_updates_same_arm(self):
        """test train accumulates statistics for same arm."""
        ps = UCB1TunedParamState(num_arms=3)
        ctx = Context()

        ps = UCB1TunedProtocol.train(ps, ctx, choice=0, reward=1.0)
        ps = UCB1TunedProtocol.train(ps, ctx, choice=0, reward=0.5)
        ps = UCB1TunedProtocol.train(ps, ctx, choice=0, reward=0.0)

        assert ps.T[0] == 3
        assert ps.mu[0] == pytest.approx(0.5)
        assert ps.round == 3

    def test_train_with_negative_reward(self):
        """test train handles negative rewards."""
        ps = UCB1TunedParamState(num_arms=3)
        ctx = Context()

        updated = UCB1TunedProtocol.train(ps, ctx, choice=0, reward=-0.5)

        assert updated.mu[0] == -0.5
        assert updated.rsq[0] == 0.25  # (-0.5)^2

    def test_train_with_large_reward(self):
        """test train handles large rewards."""
        ps = UCB1TunedParamState(num_arms=3)
        ctx = Context()

        updated = UCB1TunedProtocol.train(ps, ctx, choice=0, reward=100.0)

        assert updated.mu[0] == 100.0
        assert updated.rsq[0] == 10000.0

    def test_upper_bound_calculation_unplayed_arm(self):
        """test upper bound is infinity for unplayed arms."""
        ps = UCB1TunedParamState(num_arms=3, round=10)

        upper_bound = UCB1TunedProtocol._upper_bound(ps, 0)

        assert upper_bound == float("inf")

    def test_upper_bound_calculation_played_arm(self):
        """test upper bound calculation for played arm."""
        ps = UCB1TunedParamState(
            num_arms=3,
            mu=np.array([0.5, 0.0, 0.0]),
            T=np.array([10, 0, 0]),
            rsq=np.array([3.0, 0.0, 0.0]),
            round=10
        )

        upper_bound = UCB1TunedProtocol._upper_bound(ps, 0)

        assert isinstance(upper_bound, float)
        assert upper_bound > ps.mu[0]  # should be higher than mean

    def test_arm_var_upper_bound_unplayed_arm(self):
        """test arm variance upper bound is infinity for unplayed arms."""
        ps = UCB1TunedParamState(num_arms=3)

        var_bound = UCB1TunedProtocol._arm_var_upper_bound(ps, 0)

        assert var_bound == float("inf")

    def test_select_single_arm(self):
        """test select with single arm always returns 0."""
        ps = UCB1TunedParamState(num_arms=1)
        ctx = Context()

        arm_index = UCB1TunedProtocol.select(ps, ctx)

        assert arm_index == 0
