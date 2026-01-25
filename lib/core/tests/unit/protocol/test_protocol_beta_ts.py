"""Unit tests for BetaTSProtocol."""

import numpy as np
import pytest

from qbrixcore.protoc.stochastic.ts import BetaTSProtocol
from qbrixcore.protoc.stochastic.ts import BetaTSParamState
from qbrixcore.context import Context


class TestBetaTSParamState:
    def test_param_state_creation_minimal(self):
        """test param state creation with minimal args."""
        ps = BetaTSParamState(num_arms=3)

        assert ps.num_arms == 3
        assert ps.alpha_prior == 1.0
        assert ps.beta_prior == 1.0
        assert len(ps.alpha) == 3
        assert len(ps.beta) == 3
        assert len(ps.T) == 3
        assert np.all(ps.alpha == 1.0)
        assert np.all(ps.beta == 1.0)
        assert np.all(ps.T == 0)

    def test_param_state_creation_with_custom_priors(self):
        """test param state creation with custom priors."""
        ps = BetaTSParamState(num_arms=3, alpha_prior=2.0, beta_prior=3.0)

        assert ps.alpha_prior == 2.0
        assert ps.beta_prior == 3.0
        assert np.all(ps.alpha == 2.0)
        assert np.all(ps.beta == 3.0)

    def test_param_state_creation_with_custom_arrays(self):
        """test param state creation with pre-initialized arrays."""
        alpha = np.array([2.0, 3.0, 4.0])
        beta = np.array([1.0, 2.0, 3.0])
        T = np.array([10, 20, 30])

        ps = BetaTSParamState(num_arms=3, alpha=alpha, beta=beta, T=T)

        assert np.array_equal(ps.alpha, alpha)
        assert np.array_equal(ps.beta, beta)
        assert np.array_equal(ps.T, T)

    def test_param_state_validation_num_arms_positive(self):
        """test param state validation requires positive num_arms."""
        with pytest.raises(ValueError):
            BetaTSParamState(num_arms=0)

        with pytest.raises(ValueError):
            BetaTSParamState(num_arms=-1)

    def test_param_state_validation_alpha_prior_positive(self):
        """test param state validation requires positive alpha_prior."""
        with pytest.raises(ValueError):
            BetaTSParamState(num_arms=3, alpha_prior=0.0)

        with pytest.raises(ValueError):
            BetaTSParamState(num_arms=3, alpha_prior=-1.0)

    def test_param_state_validation_beta_prior_positive(self):
        """test param state validation requires positive beta_prior."""
        with pytest.raises(ValueError):
            BetaTSParamState(num_arms=3, beta_prior=0.0)

        with pytest.raises(ValueError):
            BetaTSParamState(num_arms=3, beta_prior=-1.0)

    def test_param_state_id_is_unique(self):
        """test param state ids are unique."""
        ps1 = BetaTSParamState(num_arms=3)
        ps2 = BetaTSParamState(num_arms=3)

        assert ps1.id != ps2.id


class TestBetaTSProtocol:
    def test_protocol_name(self):
        """test protocol has correct name."""
        assert BetaTSProtocol.name == "BetaTSProtocol"

    def test_protocol_param_state_cls(self):
        """test protocol has correct param state class."""
        assert BetaTSProtocol.param_state_cls == BetaTSParamState

    def test_init_params(self):
        """test init_params creates correct param state."""
        params = BetaTSProtocol.init_params(num_arms=4)

        assert isinstance(params, BetaTSParamState)
        assert params.num_arms == 4

    def test_init_params_with_custom_priors(self):
        """test init_params with custom priors."""
        params = BetaTSProtocol.init_params(
            num_arms=3,
            alpha_prior=2.0,
            beta_prior=3.0
        )

        assert params.alpha_prior == 2.0
        assert params.beta_prior == 3.0

    def test_select_returns_valid_arm_index(self):
        """test select returns valid arm index."""
        ps = BetaTSParamState(num_arms=5)
        ctx = Context()

        arm_index = BetaTSProtocol.select(ps, ctx)

        assert isinstance(arm_index, int)
        assert 0 <= arm_index < 5

    def test_select_with_different_params(self):
        """test select with non-uniform params."""
        # arm 0 has high success rate
        ps = BetaTSParamState(
            num_arms=3,
            alpha=np.array([100.0, 2.0, 2.0]),
            beta=np.array([2.0, 100.0, 100.0])
        )
        ctx = Context()

        # run multiple times, arm 0 should be selected most often
        selections = [BetaTSProtocol.select(ps, ctx) for _ in range(100)]
        arm_0_count = selections.count(0)

        # with these params, arm 0 should be selected the majority of times
        assert arm_0_count > 50

    def test_select_deterministic_with_seed(self):
        """test select is deterministic with fixed seed."""
        ps = BetaTSParamState(num_arms=3)
        ctx = Context()

        np.random.seed(42)
        result1 = BetaTSProtocol.select(ps, ctx)

        np.random.seed(42)
        result2 = BetaTSProtocol.select(ps, ctx)

        assert result1 == result2

    def test_train_with_success_reward(self):
        """test train updates alpha for success (reward=1)."""
        ps = BetaTSParamState(num_arms=3)
        ctx = Context()

        updated = BetaTSProtocol.train(ps, ctx, choice=1, reward=1)

        assert updated.alpha[1] == 2.0  # increased
        assert updated.beta[1] == 1.0   # unchanged
        assert updated.T[1] == 1
        # other arms unchanged
        assert updated.alpha[0] == 1.0
        assert updated.alpha[2] == 1.0

    def test_train_with_failure_reward(self):
        """test train updates beta for failure (reward=0)."""
        ps = BetaTSParamState(num_arms=3)
        ctx = Context()

        updated = BetaTSProtocol.train(ps, ctx, choice=1, reward=0)

        assert updated.alpha[1] == 1.0  # unchanged
        assert updated.beta[1] == 2.0   # increased
        assert updated.T[1] == 1

    def test_train_with_continuous_reward_above_threshold(self):
        """test train converts continuous reward > 0.5 to success."""
        ps = BetaTSParamState(num_arms=3)
        ctx = Context()

        updated = BetaTSProtocol.train(ps, ctx, choice=0, reward=0.8)

        assert updated.alpha[0] == 2.0
        assert updated.beta[0] == 1.0

    def test_train_with_continuous_reward_below_threshold(self):
        """test train converts continuous reward <= 0.5 to failure."""
        ps = BetaTSParamState(num_arms=3)
        ctx = Context()

        updated = BetaTSProtocol.train(ps, ctx, choice=0, reward=0.3)

        assert updated.alpha[0] == 1.0
        assert updated.beta[0] == 2.0

    def test_train_multiple_updates_same_arm(self):
        """test train accumulates updates for same arm."""
        ps = BetaTSParamState(num_arms=3)
        ctx = Context()

        ps = BetaTSProtocol.train(ps, ctx, choice=0, reward=1)
        ps = BetaTSProtocol.train(ps, ctx, choice=0, reward=1)
        ps = BetaTSProtocol.train(ps, ctx, choice=0, reward=0)

        assert ps.alpha[0] == 3.0  # two successes
        assert ps.beta[0] == 2.0   # one failure
        assert ps.T[0] == 3

    def test_train_does_not_mutate_original_params(self):
        """test train returns new params without mutating original."""
        ps = BetaTSParamState(num_arms=3)
        ctx = Context()
        original_alpha = ps.alpha.copy()
        original_beta = ps.beta.copy()
        original_T = ps.T.copy()

        updated = BetaTSProtocol.train(ps, ctx, choice=1, reward=1)

        # original unchanged
        assert np.array_equal(ps.alpha, original_alpha)
        assert np.array_equal(ps.beta, original_beta)
        assert np.array_equal(ps.T, original_T)
        # updated changed
        assert updated.alpha[1] == 2.0

    def test_train_with_numpy_reward(self):
        """test train handles numpy scalar reward."""
        ps = BetaTSParamState(num_arms=3)
        ctx = Context()

        updated = BetaTSProtocol.train(ps, ctx, choice=0, reward=np.float64(1.0))

        assert updated.alpha[0] == 2.0

    def test_select_single_arm(self):
        """test select with single arm always returns 0."""
        ps = BetaTSParamState(num_arms=1)
        ctx = Context()

        arm_index = BetaTSProtocol.select(ps, ctx)

        assert arm_index == 0
