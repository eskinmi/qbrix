"""Unit tests for EXP3Policy."""

import numpy as np
import pytest

from qbrixcore.policy.adversarial.exp import EXP3Policy
from qbrixcore.policy.adversarial.exp import EXP3ParamState
from qbrixcore.context import Context


class TestEXP3ParamState:
    def test_param_state_creation_minimal(self):
        """test param state creation with minimal args."""
        ps = EXP3ParamState(num_arms=3)

        assert ps.num_arms == 3
        assert ps.gamma == 0.1
        assert len(ps.w) == 3
        assert np.all(ps.w == 1.0)

    def test_param_state_creation_with_custom_gamma(self):
        """test param state creation with custom gamma."""
        ps = EXP3ParamState(num_arms=3, gamma=0.3)

        assert ps.gamma == 0.3

    def test_param_state_creation_with_custom_weights(self):
        """test param state creation with pre-initialized weights."""
        w = np.array([1.0, 2.0, 3.0])

        ps = EXP3ParamState(num_arms=3, w=w)

        assert np.array_equal(ps.w, w)

    def test_param_state_validation_num_arms_positive(self):
        """test param state validation requires positive num_arms."""
        with pytest.raises(ValueError):
            EXP3ParamState(num_arms=0)

        with pytest.raises(ValueError):
            EXP3ParamState(num_arms=-1)

    def test_param_state_validation_gamma_in_range(self):
        """test param state validation requires gamma in [0, 1]."""
        with pytest.raises(ValueError):
            EXP3ParamState(num_arms=3, gamma=-0.1)

        with pytest.raises(ValueError):
            EXP3ParamState(num_arms=3, gamma=1.5)

    def test_param_state_gamma_boundary_values(self):
        """test param state accepts gamma boundary values 0 and 1."""
        ps0 = EXP3ParamState(num_arms=3, gamma=0.0)
        ps1 = EXP3ParamState(num_arms=3, gamma=1.0)

        assert ps0.gamma == 0.0
        assert ps1.gamma == 1.0

    def test_param_state_id_is_unique(self):
        """test param state ids are unique."""
        ps1 = EXP3ParamState(num_arms=3)
        ps2 = EXP3ParamState(num_arms=3)

        assert ps1.id != ps2.id


class TestEXP3Policy:
    def test_policy_name(self):
        """test policy has correct name."""
        assert EXP3Policy.name == "EXP3Policy"

    def test_policy_param_state_cls(self):
        """test policy has correct param state class."""
        assert EXP3Policy.param_state_cls == EXP3ParamState

    def test_init_params(self):
        """test init_params creates correct param state."""
        params = EXP3Policy.init_params(num_arms=4)

        assert isinstance(params, EXP3ParamState)
        assert params.num_arms == 4

    def test_init_params_with_custom_gamma(self):
        """test init_params with custom gamma."""
        params = EXP3Policy.init_params(num_arms=3, gamma=0.2)

        assert params.gamma == 0.2

    def test_proba_with_uniform_weights(self):
        """test probability calculation with uniform weights."""
        ps = EXP3ParamState(num_arms=3, gamma=0.3)

        proba = EXP3Policy._proba(ps)

        # with uniform weights, all probabilities should be equal
        expected_prob = 1.0 / 3.0
        assert np.allclose(proba, [expected_prob, expected_prob, expected_prob])

    def test_proba_with_non_uniform_weights(self):
        """test probability calculation with non-uniform weights."""
        ps = EXP3ParamState(num_arms=3, gamma=0.1, w=np.array([1.0, 2.0, 3.0]))

        proba = EXP3Policy._proba(ps)

        # probabilities should sum to 1
        assert np.isclose(np.sum(proba), 1.0)
        # higher weight should have higher probability
        assert proba[2] > proba[1] > proba[0]

    def test_proba_sums_to_one(self):
        """test probabilities always sum to one."""
        ps = EXP3ParamState(num_arms=5, gamma=0.15, w=np.array([0.5, 1.5, 2.0, 0.3, 1.2]))

        proba = EXP3Policy._proba(ps)

        assert np.isclose(np.sum(proba), 1.0)

    def test_proba_with_gamma_zero_no_exploration(self):
        """test gamma=0 means no forced exploration."""
        ps = EXP3ParamState(num_arms=3, gamma=0.0, w=np.array([1.0, 2.0, 3.0]))

        proba = EXP3Policy._proba(ps)

        # should be purely proportional to weights
        w_sum = 6.0  # noqa
        assert np.allclose(proba, [1.0/6.0, 2.0/6.0, 3.0/6.0])

    def test_proba_with_gamma_one_uniform_exploration(self):
        """test gamma=1 means fully uniform exploration."""
        ps = EXP3ParamState(num_arms=3, gamma=1.0, w=np.array([100.0, 1.0, 1.0]))

        proba = EXP3Policy._proba(ps)

        # should be uniform regardless of weights
        assert np.allclose(proba, [1.0/3.0, 1.0/3.0, 1.0/3.0])

    def test_select_returns_valid_arm_index(self):
        """test select returns valid arm index."""
        ps = EXP3ParamState(num_arms=5)
        ctx = Context()

        arm_index = EXP3Policy.select(ps, ctx)

        assert isinstance(arm_index, int)
        assert 0 <= arm_index < 5

    def test_select_with_uniform_weights_samples_all_arms(self):
        """test select with uniform weights samples all arms over time."""
        ps = EXP3ParamState(num_arms=3, gamma=0.1)
        ctx = Context()

        # run multiple selections
        selections = [EXP3Policy.select(ps, ctx) for _ in range(100)]

        # all arms should be selected at least once
        assert len(set(selections)) == 3

    def test_select_deterministic_with_seed(self):
        """test select is deterministic with fixed seed."""
        ps = EXP3ParamState(num_arms=3)
        ctx = Context()

        np.random.seed(42)
        result1 = EXP3Policy.select(ps, ctx)

        np.random.seed(42)
        result2 = EXP3Policy.select(ps, ctx)

        assert result1 == result2

    def test_select_favors_higher_weight_arm(self):
        """test select favors arm with higher weight."""
        ps = EXP3ParamState(
            num_arms=3,
            gamma=0.0,  # no forced exploration
            w=np.array([1.0, 1.0, 100.0])
        )
        ctx = Context()

        # run many selections
        selections = [EXP3Policy.select(ps, ctx) for _ in range(100)]
        arm_2_count = selections.count(2)

        # arm 2 should be selected most often
        assert arm_2_count > 90

    def test_train_updates_weights_with_positive_reward(self):
        """test train updates weights correctly with positive reward."""
        ps = EXP3ParamState(num_arms=3, gamma=0.1)
        ctx = Context()
        original_w = ps.w.copy()

        updated = EXP3Policy.train(ps, ctx, choice=1, reward=1.0)

        # weights should be updated
        assert not np.array_equal(updated.w, original_w)
        # weights should sum to 1 (normalized)
        assert np.isclose(np.sum(updated.w), 1.0)

    def test_train_updates_weights_with_zero_reward(self):
        """test train handles zero reward."""
        ps = EXP3ParamState(num_arms=3, gamma=0.1)
        ctx = Context()

        updated = EXP3Policy.train(ps, ctx, choice=1, reward=0.0)

        # weights should still be normalized
        assert np.isclose(np.sum(updated.w), 1.0)

    def test_train_importance_weighted_reward(self):
        """test train uses importance-weighted reward estimate."""
        ps = EXP3ParamState(num_arms=3, gamma=0.1)
        ctx = Context()

        updated = EXP3Policy.train(ps, ctx, choice=0, reward=1.0)

        # after normalization, chosen arm should have higher relative weight
        # compared to other arms (since it got positive reward)
        assert updated.w[0] > updated.w[1]
        assert updated.w[0] > updated.w[2]
        # weights should still sum to 1
        assert np.isclose(np.sum(updated.w), 1.0)

    def test_train_does_not_mutate_original_params(self):
        """test train returns new params without mutating original."""
        ps = EXP3ParamState(num_arms=3)
        ctx = Context()
        original_w = ps.w.copy()

        updated = EXP3Policy.train(ps, ctx, choice=1, reward=1.0)

        # original unchanged
        assert np.array_equal(ps.w, original_w)
        # updated changed
        assert not np.array_equal(updated.w, ps.w)

    def test_train_multiple_updates_same_arm(self):
        """test train accumulates updates for same arm."""
        ps = EXP3ParamState(num_arms=3, gamma=0.1)
        ctx = Context()

        ps = EXP3Policy.train(ps, ctx, choice=0, reward=1.0)
        ps = EXP3Policy.train(ps, ctx, choice=0, reward=1.0)
        ps = EXP3Policy.train(ps, ctx, choice=0, reward=1.0)

        # after multiple positive rewards, arm 0 should have higher weight
        assert ps.w[0] > ps.w[1]
        assert ps.w[0] > ps.w[2]

    def test_train_with_negative_reward(self):
        """test train handles negative rewards."""
        ps = EXP3ParamState(num_arms=3, gamma=0.1)
        ctx = Context()

        updated = EXP3Policy.train(ps, ctx, choice=0, reward=-1.0)

        # negative reward should decrease relative weight
        relative_change = updated.w[0] / ps.w[0]
        assert relative_change < 1.0

    def test_train_with_numpy_reward(self):
        """test train handles numpy scalar reward."""
        ps = EXP3ParamState(num_arms=3)
        ctx = Context()

        updated = EXP3Policy.train(ps, ctx, choice=0, reward=np.float64(1.0))

        assert updated is not None
        assert np.isclose(np.sum(updated.w), 1.0)

    def test_select_single_arm(self):
        """test select with single arm always returns 0."""
        ps = EXP3ParamState(num_arms=1)
        ctx = Context()

        arm_index = EXP3Policy.select(ps, ctx)

        assert arm_index == 0

    def test_integration_select_and_train(self):
        """test integration of select and train workflow."""
        ps = EXP3ParamState(num_arms=3, gamma=0.2)
        ctx = Context()

        # select arm
        arm = EXP3Policy.select(ps, ctx)
        assert 0 <= arm < 3

        # train on selection with positive reward
        updated_ps = EXP3Policy.train(ps, ctx, choice=arm, reward=1.0)

        # weights should be normalized
        assert np.isclose(np.sum(updated_ps.w), 1.0)

        # select again with updated params
        arm2 = EXP3Policy.select(updated_ps, ctx)
        assert 0 <= arm2 < 3

    def test_train_normalization_prevents_overflow(self):
        """test train normalization prevents weight overflow."""
        ps = EXP3ParamState(num_arms=3, gamma=0.1)
        ctx = Context()

        # perform many updates with high rewards
        for _ in range(100):
            ps = EXP3Policy.train(ps, ctx, choice=0, reward=10.0)

        # weights should still be normalized
        assert np.isclose(np.sum(ps.w), 1.0)
        # weights should not be infinite or nan
        assert np.all(np.isfinite(ps.w))

    def test_exploration_parameter_affects_distribution(self):
        """test gamma parameter affects exploration behavior."""
        w = np.array([10.0, 1.0, 1.0])

        ps_low_gamma = EXP3ParamState(num_arms=3, gamma=0.01, w=w.copy())
        ps_high_gamma = EXP3ParamState(num_arms=3, gamma=0.5, w=w.copy())

        proba_low = EXP3Policy._proba(ps_low_gamma)
        proba_high = EXP3Policy._proba(ps_high_gamma)

        # with higher gamma, probabilities should be more uniform
        # (less difference between max and min probabilities)
        diff_low = np.max(proba_low) - np.min(proba_low)  # noqa
        diff_high = np.max(proba_high) - np.min(proba_high)  # noqa

        assert diff_high < diff_low
