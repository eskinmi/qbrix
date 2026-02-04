"""Unit tests for GaussianTSPolicy."""

import numpy as np
import pytest

from qbrixcore.policy.stochastic.ts import GaussianTSPolicy
from qbrixcore.policy.stochastic.ts import GaussianTSParamState
from qbrixcore.context import Context


class TestGaussianTSParamState:
    def test_param_state_creation_minimal(self):
        """test param state creation with minimal args."""
        ps = GaussianTSParamState(num_arms=3)

        assert ps.num_arms == 3
        assert ps.prior_mean == 0.0
        assert ps.prior_precision == 1.0
        assert ps.noise_precision == 1.0
        assert len(ps.posterior_mean) == 3
        assert len(ps.posterior_precision) == 3
        assert len(ps.T) == 3
        assert np.all(ps.posterior_mean == 0.0)
        assert np.all(ps.posterior_precision == 1.0)
        assert np.all(ps.T == 0)

    def test_param_state_creation_with_custom_priors(self):
        """test param state creation with custom priors."""
        ps = GaussianTSParamState(
            num_arms=3,
            prior_mean=5.0,
            prior_precision=2.0,
            noise_precision=3.0
        )

        assert ps.prior_mean == 5.0
        assert ps.prior_precision == 2.0
        assert ps.noise_precision == 3.0
        assert np.all(ps.posterior_mean == 5.0)
        assert np.all(ps.posterior_precision == 2.0)

    def test_param_state_creation_with_custom_arrays(self):
        """test param state creation with pre-initialized arrays."""
        posterior_mean = np.array([0.5, 0.6, 0.7])
        posterior_precision = np.array([1.5, 2.0, 2.5])
        T = np.array([10, 20, 30])

        ps = GaussianTSParamState(
            num_arms=3,
            posterior_mean=posterior_mean,
            posterior_precision=posterior_precision,
            T=T
        )

        assert np.array_equal(ps.posterior_mean, posterior_mean)
        assert np.array_equal(ps.posterior_precision, posterior_precision)
        assert np.array_equal(ps.T, T)

    def test_param_state_validation_num_arms_positive(self):
        """test param state validation requires positive num_arms."""
        with pytest.raises(ValueError):
            GaussianTSParamState(num_arms=0)

    def test_param_state_validation_prior_precision_positive(self):
        """test param state validation requires positive prior_precision."""
        with pytest.raises(ValueError):
            GaussianTSParamState(num_arms=3, prior_precision=0.0)

    def test_param_state_validation_noise_precision_positive(self):
        """test param state validation requires positive noise_precision."""
        with pytest.raises(ValueError):
            GaussianTSParamState(num_arms=3, noise_precision=0.0)


class TestGaussianTSPolicy:
    def test_policy_name(self):
        """test policy has correct name."""
        assert GaussianTSPolicy.name == "GaussianTSPolicy"

    def test_policy_param_state_cls(self):
        """test policy has correct param state class."""
        assert GaussianTSPolicy.param_state_cls == GaussianTSParamState

    def test_init_params(self):
        """test init_params creates correct param state."""
        params = GaussianTSPolicy.init_params(num_arms=4)

        assert isinstance(params, GaussianTSParamState)
        assert params.num_arms == 4

    def test_init_params_with_custom_priors(self):
        """test init_params with custom priors."""
        params = GaussianTSPolicy.init_params(
            num_arms=3,
            prior_mean=2.0,
            prior_precision=1.5
        )

        assert params.prior_mean == 2.0
        assert params.prior_precision == 1.5

    def test_select_returns_valid_arm_index(self):
        """test select returns valid arm index."""
        ps = GaussianTSParamState(num_arms=5)
        ctx = Context()

        arm_index = GaussianTSPolicy.select(ps, ctx)

        assert isinstance(arm_index, int)
        assert 0 <= arm_index < 5

    def test_select_with_different_means(self):
        """test select with non-uniform posterior means."""
        # arm 0 has high posterior mean
        ps = GaussianTSParamState(
            num_arms=3,
            posterior_mean=np.array([10.0, 0.5, 0.5]),
            posterior_precision=np.array([100.0, 100.0, 100.0])  # high precision = low variance
        )
        ctx = Context()

        # run multiple times, arm 0 should be selected most often
        selections = [GaussianTSPolicy.select(ps, ctx) for _ in range(100)]
        arm_0_count = selections.count(0)

        # with high mean and low variance, arm 0 should be selected the majority of times
        assert arm_0_count > 50

    def test_select_deterministic_with_seed(self):
        """test select is deterministic with fixed seed."""
        ps = GaussianTSParamState(num_arms=3)
        ctx = Context()

        np.random.seed(42)
        result1 = GaussianTSPolicy.select(ps, ctx)

        np.random.seed(42)
        result2 = GaussianTSPolicy.select(ps, ctx)

        assert result1 == result2

    def test_train_updates_posterior_mean(self):
        """test train updates posterior mean."""
        ps = GaussianTSParamState(
            num_arms=3,
            prior_mean=0.0,
            prior_precision=1.0,
            noise_precision=1.0
        )
        ctx = Context()

        updated = GaussianTSPolicy.train(ps, ctx, choice=1, reward=2.0)

        # with equal precision, new mean is average of prior and observation
        expected_mean = (1.0 * 0.0 + 1.0 * 2.0) / 2.0
        assert updated.posterior_mean[1] == pytest.approx(expected_mean)
        # other arms unchanged
        assert updated.posterior_mean[0] == 0.0
        assert updated.posterior_mean[2] == 0.0

    def test_train_updates_posterior_precision(self):
        """test train updates posterior precision."""
        ps = GaussianTSParamState(
            num_arms=3,
            prior_precision=1.0,
            noise_precision=1.0
        )
        ctx = Context()

        updated = GaussianTSPolicy.train(ps, ctx, choice=1, reward=1.0)

        # precision increases by noise_precision
        assert updated.posterior_precision[1] == 2.0
        assert updated.T[1] == 1
        # other arms unchanged
        assert updated.posterior_precision[0] == 1.0

    def test_train_multiple_updates_same_arm(self):
        """test train accumulates updates for same arm."""
        ps = GaussianTSParamState(num_arms=3)
        ctx = Context()

        ps = GaussianTSPolicy.train(ps, ctx, choice=0, reward=1.0)
        ps = GaussianTSPolicy.train(ps, ctx, choice=0, reward=2.0)
        ps = GaussianTSPolicy.train(ps, ctx, choice=0, reward=3.0)

        assert ps.T[0] == 3
        assert ps.posterior_precision[0] == 4.0  # 1.0 + 3 * 1.0
        assert ps.posterior_mean[0] > 0.0  # should be positive

    def test_train_does_not_mutate_original_params(self):
        """test train returns new params without mutating original."""
        ps = GaussianTSParamState(num_arms=3)
        ctx = Context()
        original_mean = ps.posterior_mean.copy()
        original_precision = ps.posterior_precision.copy()
        original_T = ps.T.copy()

        updated = GaussianTSPolicy.train(ps, ctx, choice=1, reward=1.0)

        # original unchanged
        assert np.array_equal(ps.posterior_mean, original_mean)
        assert np.array_equal(ps.posterior_precision, original_precision)
        assert np.array_equal(ps.T, original_T)
        # updated changed
        assert updated.posterior_mean[1] != ps.posterior_mean[1]

    def test_train_with_negative_reward(self):
        """test train handles negative rewards."""
        ps = GaussianTSParamState(num_arms=3)
        ctx = Context()

        updated = GaussianTSPolicy.train(ps, ctx, choice=0, reward=-2.0)

        assert updated.posterior_mean[0] < 0.0

    def test_train_with_large_reward(self):
        """test train handles large rewards."""
        ps = GaussianTSParamState(num_arms=3)
        ctx = Context()

        updated = GaussianTSPolicy.train(ps, ctx, choice=0, reward=100.0)

        assert updated.posterior_mean[0] > 0.0

    def test_train_with_high_noise_precision(self):
        """test train with high noise precision trusts observation more."""
        ps = GaussianTSParamState(
            num_arms=3,
            prior_mean=0.0,
            prior_precision=1.0,
            noise_precision=10.0  # high confidence in observations
        )
        ctx = Context()

        updated = GaussianTSPolicy.train(ps, ctx, choice=0, reward=5.0)

        # with high noise precision, posterior mean should be close to observation
        assert updated.posterior_mean[0] > 4.0

    def test_train_with_low_noise_precision(self):
        """test train with low noise precision trusts prior more."""
        ps = GaussianTSParamState(
            num_arms=3,
            prior_mean=0.0,
            prior_precision=10.0,  # high confidence in prior
            noise_precision=0.1    # low confidence in observations
        )
        ctx = Context()

        updated = GaussianTSPolicy.train(ps, ctx, choice=0, reward=5.0)

        # with low noise precision, posterior mean should stay close to prior
        assert abs(updated.posterior_mean[0]) < 1.0

    def test_train_with_numpy_reward(self):
        """test train handles numpy scalar reward."""
        ps = GaussianTSParamState(num_arms=3)
        ctx = Context()

        updated = GaussianTSPolicy.train(ps, ctx, choice=0, reward=np.float64(2.0))

        assert updated.posterior_mean[0] > 0.0

    def test_select_single_arm(self):
        """test select with single arm always returns 0."""
        ps = GaussianTSParamState(num_arms=1)
        ctx = Context()

        arm_index = GaussianTSPolicy.select(ps, ctx)

        assert arm_index == 0

    def test_train_convergence(self):
        """test train converges posterior mean towards true mean with repeated observations."""
        ps = GaussianTSParamState(num_arms=1, prior_mean=0.0)
        ctx = Context()
        true_mean = 3.5

        # train with many observations from true mean
        for _ in range(100):
            ps = GaussianTSPolicy.train(ps, ctx, choice=0, reward=true_mean)

        # posterior mean should converge to true mean
        assert ps.posterior_mean[0] == pytest.approx(true_mean, rel=0.01)
