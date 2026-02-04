"""Unit tests for FPLPolicy."""

import numpy as np
import pytest

from qbrixcore.policy.adversarial.fpl import FPLPolicy
from qbrixcore.policy.adversarial.fpl import FPLParamState
from qbrixcore.context import Context


class TestFPLParamState:
    def test_param_state_creation_minimal(self):
        """test param state creation with minimal args."""
        ps = FPLParamState(num_arms=3)

        assert ps.num_arms == 3
        assert ps.eta == 5.0
        assert len(ps.r) == 3
        assert np.all(ps.r == 0.0)

    def test_param_state_creation_with_custom_eta(self):
        """test param state creation with custom eta."""
        ps = FPLParamState(num_arms=3, eta=10.0)

        assert ps.eta == 10.0

    def test_param_state_creation_with_custom_rewards(self):
        """test param state creation with pre-initialized rewards."""
        r = np.array([1.5, 2.5, 3.5])

        ps = FPLParamState(num_arms=3, r=r)

        assert np.array_equal(ps.r, r)

    def test_param_state_validation_num_arms_positive(self):
        """test param state validation requires positive num_arms."""
        with pytest.raises(ValueError):
            FPLParamState(num_arms=0)

        with pytest.raises(ValueError):
            FPLParamState(num_arms=-1)

    def test_param_state_validation_eta_positive(self):
        """test param state validation requires positive eta."""
        with pytest.raises(ValueError):
            FPLParamState(num_arms=3, eta=0.0)

        with pytest.raises(ValueError):
            FPLParamState(num_arms=3, eta=-1.0)

    def test_param_state_id_is_unique(self):
        """test param state ids are unique."""
        ps1 = FPLParamState(num_arms=3)
        ps2 = FPLParamState(num_arms=3)

        assert ps1.id != ps2.id


class TestFPLPolicy:
    def test_policy_name(self):
        """test policy has correct name."""
        assert FPLPolicy.name == "FPLPolicy"

    def test_policy_param_state_cls(self):
        """test policy has correct param state class."""
        assert FPLPolicy.param_state_cls == FPLParamState

    def test_init_params(self):
        """test init_params creates correct param state."""
        params = FPLPolicy.init_params(num_arms=4)

        assert isinstance(params, FPLParamState)
        assert params.num_arms == 4

    def test_init_params_with_custom_eta(self):
        """test init_params with custom eta."""
        params = FPLPolicy.init_params(num_arms=3, eta=8.0)

        assert params.eta == 8.0

    def test_select_returns_valid_arm_index(self):
        """test select returns valid arm index."""
        ps = FPLParamState(num_arms=5)
        ctx = Context()

        arm_index = FPLPolicy.select(ps, ctx)

        assert isinstance(arm_index, int)
        assert 0 <= arm_index < 5

    def test_select_with_zero_rewards_random_exploration(self):
        """test select with zero cumulative rewards explores randomly."""
        ps = FPLParamState(num_arms=3, eta=5.0)
        ctx = Context()

        # run multiple selections
        selections = [FPLPolicy.select(ps, ctx) for _ in range(100)]

        # with zero rewards and noise, all arms should be selected
        assert len(set(selections)) == 3

    def test_select_deterministic_with_seed(self):
        """test select is deterministic with fixed seed."""
        ps = FPLParamState(num_arms=3)
        ctx = Context()

        np.random.seed(42)
        result1 = FPLPolicy.select(ps, ctx)

        np.random.seed(42)
        result2 = FPLPolicy.select(ps, ctx)

        assert result1 == result2

    def test_select_prefers_arm_with_higher_cumulative_reward(self):
        """test select prefers arm with higher cumulative reward."""
        ps = FPLParamState(
            num_arms=3,
            eta=0.1,  # low noise to emphasize reward differences
            r=np.array([100.0, 1.0, 1.0])
        )
        ctx = Context()

        # run multiple selections
        selections = [FPLPolicy.select(ps, ctx) for _ in range(100)]
        arm_0_count = selections.count(0)

        # arm 0 should be selected most often due to high cumulative reward
        assert arm_0_count > 50

    def test_select_single_arm(self):
        """test select with single arm always returns 0."""
        ps = FPLParamState(num_arms=1)
        ctx = Context()

        arm_index = FPLPolicy.select(ps, ctx)

        assert arm_index == 0

    def test_train_updates_cumulative_reward_for_chosen_arm(self):
        """test train updates cumulative reward for chosen arm."""
        ps = FPLParamState(num_arms=3)
        ctx = Context()

        updated = FPLPolicy.train(ps, ctx, choice=1, reward=2.5)

        assert updated.r[1] == 2.5
        # other arms unchanged
        assert updated.r[0] == 0.0
        assert updated.r[2] == 0.0

    def test_train_with_negative_reward(self):
        """test train handles negative rewards."""
        ps = FPLParamState(num_arms=3)
        ctx = Context()

        updated = FPLPolicy.train(ps, ctx, choice=0, reward=-1.5)

        assert updated.r[0] == -1.5

    def test_train_accumulates_rewards(self):
        """test train accumulates rewards for same arm."""
        ps = FPLParamState(num_arms=3)
        ctx = Context()

        ps = FPLPolicy.train(ps, ctx, choice=0, reward=1.0)
        ps = FPLPolicy.train(ps, ctx, choice=0, reward=2.0)
        ps = FPLPolicy.train(ps, ctx, choice=0, reward=0.5)

        assert ps.r[0] == 3.5

    def test_train_does_not_mutate_original_params(self):
        """test train returns new params without mutating original."""
        ps = FPLParamState(num_arms=3)
        ctx = Context()
        original_r = ps.r.copy()

        updated = FPLPolicy.train(ps, ctx, choice=1, reward=1.0)

        # original unchanged
        assert np.array_equal(ps.r, original_r)
        # updated changed
        assert updated.r[1] == 1.0
        assert ps.r[1] == 0.0

    def test_train_with_numpy_reward(self):
        """test train handles numpy scalar reward."""
        ps = FPLParamState(num_arms=3)
        ctx = Context()

        updated = FPLPolicy.train(ps, ctx, choice=0, reward=np.float64(1.5))

        assert updated.r[0] == 1.5

    def test_train_multiple_arms(self):
        """test train can update different arms independently."""
        ps = FPLParamState(num_arms=3)
        ctx = Context()

        ps = FPLPolicy.train(ps, ctx, choice=0, reward=1.0)
        ps = FPLPolicy.train(ps, ctx, choice=1, reward=2.0)
        ps = FPLPolicy.train(ps, ctx, choice=2, reward=0.5)

        assert ps.r[0] == 1.0
        assert ps.r[1] == 2.0
        assert ps.r[2] == 0.5

    def test_integration_select_and_train(self):
        """test integration of select and train workflow."""
        ps = FPLParamState(num_arms=3, eta=5.0)
        ctx = Context()

        # select arm
        arm = FPLPolicy.select(ps, ctx)
        assert 0 <= arm < 3

        # train on selection
        updated_ps = FPLPolicy.train(ps, ctx, choice=arm, reward=1.0)
        assert updated_ps.r[arm] == 1.0

        # select again with updated params
        arm2 = FPLPolicy.select(updated_ps, ctx)
        assert 0 <= arm2 < 3

    def test_eta_affects_exploration(self):
        """test eta parameter affects exploration behavior."""
        r = np.array([100.0, 0.0, 0.0])

        ps_low_eta = FPLParamState(num_arms=3, eta=0.1, r=r.copy())
        ps_high_eta = FPLParamState(num_arms=3, eta=50.0, r=r.copy())
        ctx = Context()

        # with low eta (low noise), arm 0 should be selected very consistently
        selections_low = [FPLPolicy.select(ps_low_eta, ctx) for _ in range(100)]
        arm_0_count_low = selections_low.count(0)

        # with high eta (high noise), more exploration expected
        selections_high = [FPLPolicy.select(ps_high_eta, ctx) for _ in range(100)]
        arm_0_count_high = selections_high.count(0)

        # low eta should have higher consistency on best arm
        assert arm_0_count_low > arm_0_count_high

    def test_perturbation_uses_exponential_distribution(self):
        """test select uses exponential noise for perturbation."""
        ps = FPLParamState(num_arms=3, eta=5.0, r=np.zeros(3))
        ctx = Context()

        # with zero rewards, selection is purely based on exponential noise
        # run many selections and verify distribution properties
        np.random.seed(42)
        selections = [FPLPolicy.select(ps, ctx) for _ in range(1000)]

        # all arms should be selected (exploratory behavior)
        assert len(set(selections)) == 3
        # no single arm should dominate too much (due to randomness)
        for arm in range(3):
            count = selections.count(arm)
            # each arm should be selected between 20% and 45% of the time
            assert 200 < count < 450
