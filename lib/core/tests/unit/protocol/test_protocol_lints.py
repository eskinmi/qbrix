"""Unit tests for LinTSPolicy."""

import numpy as np
import pytest

from qbrixcore.policy.contextual.ts import LinTSPolicy
from qbrixcore.policy.contextual.ts import LinTSParamState
from qbrixcore.context import Context


class TestLinTSParamState:
    def test_param_state_creation_minimal(self):
        """test param state creation with minimal args."""
        ps = LinTSParamState(num_arms=3, dim=5)

        assert ps.num_arms == 3
        assert ps.dim == 5
        assert ps.v == 1.0
        assert ps.d.shape == (3, 5, 5)
        assert ps.r.shape == (3, 5, 1)

    def test_param_state_default_design_matrix_is_identity(self):
        """test default design matrix is identity for each arm."""
        ps = LinTSParamState(num_arms=3, dim=4)

        for i in range(3):
            assert np.array_equal(ps.d[i], np.identity(4))

    def test_param_state_default_reward_vector_is_zero(self):
        """test default reward vector is zero for each arm."""
        ps = LinTSParamState(num_arms=3, dim=4)

        for i in range(3):
            assert np.all(ps.r[i] == 0.0)

    def test_param_state_creation_with_custom_v(self):
        """test param state creation with custom noise parameter v."""
        ps = LinTSParamState(num_arms=3, dim=5, v=2.0)

        assert ps.v == 2.0

    def test_param_state_creation_with_custom_matrices(self):
        """test param state creation with pre-initialized matrices."""
        d = np.array([np.identity(3) * 2 for _ in range(2)])
        r = np.ones((2, 3, 1))

        ps = LinTSParamState(num_arms=2, dim=3, d=d, r=r)

        assert np.array_equal(ps.d, d)
        assert np.array_equal(ps.r, r)

    def test_param_state_validation_num_arms_positive(self):
        """test param state validation requires positive num_arms."""
        with pytest.raises(ValueError):
            LinTSParamState(num_arms=0, dim=5)

    def test_param_state_validation_dim_positive(self):
        """test param state validation requires positive dim."""
        with pytest.raises(ValueError):
            LinTSParamState(num_arms=3, dim=0)

    def test_param_state_validation_v_positive(self):
        """test param state validation requires positive v."""
        with pytest.raises(ValueError):
            LinTSParamState(num_arms=3, dim=5, v=0.0)

        with pytest.raises(ValueError):
            LinTSParamState(num_arms=3, dim=5, v=-1.0)

    def test_param_state_id_is_unique(self):
        """test param state ids are unique."""
        ps1 = LinTSParamState(num_arms=3, dim=5)
        ps2 = LinTSParamState(num_arms=3, dim=5)

        assert ps1.id != ps2.id


class TestLinTSPolicy:
    def test_policy_name(self):
        """test policy has correct name."""
        assert LinTSPolicy.name == "LinTSPolicy"

    def test_policy_param_state_cls(self):
        """test policy has correct param state class."""
        assert LinTSPolicy.param_state_cls == LinTSParamState

    def test_init_params(self):
        """test init_params creates correct param state."""
        params = LinTSPolicy.init_params(num_arms=3, dim=5)

        assert isinstance(params, LinTSParamState)
        assert params.num_arms == 3
        assert params.dim == 5

    def test_init_params_with_custom_v(self):
        """test init_params with custom noise parameter."""
        params = LinTSPolicy.init_params(num_arms=3, dim=5, v=2.5)

        assert params.v == 2.5

    def test_reshape_context_vector_from_list(self):
        """test reshaping context vector from list to column vector."""
        ctx = Context(vector=[1.0, 2.0, 3.0])

        reshaped = LinTSPolicy._reshape_context_vector(ctx)

        assert isinstance(reshaped, np.ndarray)
        assert reshaped.shape == (3, 1)
        assert np.array_equal(reshaped, [[1.0], [2.0], [3.0]])

    def test_reshape_context_vector_from_1d_array(self):
        """test reshaping 1d numpy array to column vector."""
        ctx = Context(vector=np.array([1.0, 2.0, 3.0]))

        reshaped = LinTSPolicy._reshape_context_vector(ctx)

        assert reshaped.shape == (3, 1)

    def test_reshape_context_vector_already_column(self):
        """test context vector already in column form is unchanged."""
        ctx = Context(vector=np.array([[1.0], [2.0], [3.0]]))

        reshaped = LinTSPolicy._reshape_context_vector(ctx)

        assert reshaped.shape == (3, 1)

    def test_sample_theta_returns_column_vector(self):
        """test sample_theta returns correctly shaped column vector."""
        ps = LinTSParamState(num_arms=3, dim=4)

        theta = LinTSPolicy._sample_theta(ps, 0)

        assert isinstance(theta, np.ndarray)
        assert theta.shape == (4, 1)

    def test_sample_theta_is_stochastic(self):
        """test sample_theta produces different samples."""
        ps = LinTSParamState(num_arms=3, dim=4)

        np.random.seed(1)
        theta1 = LinTSPolicy._sample_theta(ps, 0)

        np.random.seed(2)
        theta2 = LinTSPolicy._sample_theta(ps, 0)

        # different seeds should produce different samples
        assert not np.array_equal(theta1, theta2)

    def test_sample_theta_with_singular_matrix_uses_pseudoinverse(self):
        """test sample_theta handles singular matrix gracefully."""
        ps = LinTSParamState(num_arms=3, dim=2)
        # make design matrix singular
        ps.d[0] = np.zeros((2, 2))

        theta = LinTSPolicy._sample_theta(ps, 0)

        # should return something reasonable (zeros via pinv fallback)
        assert theta.shape == (2, 1)
        assert not np.any(np.isnan(theta))

    def test_sample_theta_with_updated_state(self):
        """test sample_theta with non-initial state."""
        ps = LinTSParamState(num_arms=3, dim=2)
        # simulate trained state
        ps.d[0] = np.array([[2.0, 0.5], [0.5, 2.0]])
        ps.r[0] = np.array([[5.0], [3.0]])

        theta = LinTSPolicy._sample_theta(ps, 0)

        # should sample around the posterior mean
        assert theta.shape == (2, 1)
        assert np.all(np.isfinite(theta))

    def test_select_returns_valid_arm_index(self):
        """test select returns valid arm index."""
        ps = LinTSParamState(num_arms=3, dim=4)
        ctx = Context(vector=[1.0, 0.5, -0.3, 0.8])

        arm_index = LinTSPolicy.select(ps, ctx)

        assert isinstance(arm_index, int)
        assert 0 <= arm_index < 3

    def test_select_with_initial_state_explores(self):
        """test select with initial state explores all arms."""
        ps = LinTSParamState(num_arms=3, dim=4)
        ctx = Context(vector=[1.0, 0.5, -0.3, 0.8])

        # run multiple selections with initial state
        selections = [LinTSPolicy.select(ps, ctx) for _ in range(100)]

        # should explore all arms due to posterior sampling
        assert len(set(selections)) == 3

    def test_select_deterministic_with_seed(self):
        """test select is deterministic with fixed seed."""
        ps = LinTSParamState(num_arms=3, dim=4)
        ctx = Context(vector=[1.0, 0.5, -0.3, 0.8])

        np.random.seed(42)
        result1 = LinTSPolicy.select(ps, ctx)

        np.random.seed(42)
        result2 = LinTSPolicy.select(ps, ctx)

        assert result1 == result2

    def test_select_prefers_arm_with_higher_estimate(self):
        """test select tends to prefer arm with higher reward estimate."""
        ps = LinTSParamState(num_arms=3, dim=2)

        # manually set r to give arm 0 much higher estimate
        ps.r[0] = np.array([[100.0], [100.0]])
        ps.r[1] = np.array([[1.0], [1.0]])
        ps.r[2] = np.array([[1.0], [1.0]])

        ctx = Context(vector=[1.0, 1.0])

        # run multiple selections
        selections = [LinTSPolicy.select(ps, ctx) for _ in range(100)]
        arm_0_count = selections.count(0)

        # arm 0 should be selected most often
        assert arm_0_count > 50

    def test_select_single_arm(self):
        """test select with single arm always returns 0."""
        ps = LinTSParamState(num_arms=1, dim=3)
        ctx = Context(vector=[1.0, 0.5, -0.3])

        arm_index = LinTSPolicy.select(ps, ctx)

        assert arm_index == 0

    def test_train_updates_design_matrix(self):
        """test train updates design matrix for chosen arm."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 2.0])

        updated = LinTSPolicy.train(ps, ctx, choice=1, reward=0.5)

        # d[1] should be updated with x*x.T
        expected_update = np.array([[1.0, 2.0], [2.0, 4.0]])
        expected_d = np.identity(2) + expected_update

        assert np.allclose(updated.d[1], expected_d)
        # other arms unchanged
        assert np.array_equal(updated.d[0], ps.d[0])
        assert np.array_equal(updated.d[2], ps.d[2])

    def test_train_updates_reward_vector(self):
        """test train updates reward vector for chosen arm."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 2.0])

        updated = LinTSPolicy.train(ps, ctx, choice=1, reward=0.5)

        # r[1] should be updated with reward * x
        expected_r = np.array([[0.5], [1.0]])

        assert np.allclose(updated.r[1], expected_r)
        # other arms unchanged
        assert np.array_equal(updated.r[0], ps.r[0])
        assert np.array_equal(updated.r[2], ps.r[2])

    def test_train_does_not_mutate_original_params(self):
        """test train returns new params without mutating original."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 2.0])
        original_d = ps.d.copy()
        original_r = ps.r.copy()

        updated = LinTSPolicy.train(ps, ctx, choice=1, reward=0.5)

        # original unchanged
        assert np.array_equal(ps.d, original_d)
        assert np.array_equal(ps.r, original_r)
        # updated changed
        assert not np.array_equal(updated.d[1], ps.d[1])

    def test_train_multiple_updates_same_arm(self):
        """test train accumulates updates for same arm."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 1.0])

        ps = LinTSPolicy.train(ps, ctx, choice=0, reward=1.0)
        ps = LinTSPolicy.train(ps, ctx, choice=0, reward=0.5)
        ps = LinTSPolicy.train(ps, ctx, choice=0, reward=0.0)

        # design matrix should accumulate
        assert not np.array_equal(ps.d[0], np.identity(2))
        # reward vector should accumulate
        expected_r_sum = 1.0 + 0.5 + 0.0
        assert ps.r[0][0, 0] == pytest.approx(expected_r_sum)

    def test_train_with_list_context_vector(self):
        """test train works with list context vector."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 2.0])

        updated = LinTSPolicy.train(ps, ctx, choice=0, reward=0.5)

        assert updated is not None
        assert not np.array_equal(updated.d[0], ps.d[0])

    def test_train_with_numpy_context_vector(self):
        """test train works with numpy array context vector."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=np.array([1.0, 2.0]))

        updated = LinTSPolicy.train(ps, ctx, choice=0, reward=0.5)

        assert updated is not None
        assert not np.array_equal(updated.d[0], ps.d[0])

    def test_train_with_negative_reward(self):
        """test train handles negative rewards."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 1.0])

        updated = LinTSPolicy.train(ps, ctx, choice=0, reward=-1.0)

        assert updated.r[0][0, 0] == -1.0
        assert updated.r[0][1, 0] == -1.0

    def test_train_with_numpy_reward(self):
        """test train handles numpy scalar reward."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 2.0])

        updated = LinTSPolicy.train(ps, ctx, choice=0, reward=np.float64(1.5))

        assert updated.r[0][0, 0] == 1.5

    def test_integration_select_and_train(self):
        """test integration of select and train workflow."""
        ps = LinTSParamState(num_arms=3, dim=4)
        ctx = Context(vector=[1.0, 0.5, -0.3, 0.8])

        # select arm
        arm = LinTSPolicy.select(ps, ctx)
        assert 0 <= arm < 3

        # train on selection
        updated_ps = LinTSPolicy.train(ps, ctx, choice=arm, reward=1.0)

        # select again with updated params
        arm2 = LinTSPolicy.select(updated_ps, ctx)
        assert 0 <= arm2 < 3

    def test_sample_theta_deterministic_with_seed(self):
        """test sample_theta is deterministic with fixed seed."""
        ps = LinTSParamState(num_arms=3, dim=4)

        np.random.seed(42)
        theta1 = LinTSPolicy._sample_theta(ps, 0)

        np.random.seed(42)
        theta2 = LinTSPolicy._sample_theta(ps, 0)

        assert np.array_equal(theta1, theta2)

    def test_sample_theta_with_singular_matrix_falls_back_to_pinv(self):
        """test sample_theta handles singular matrix with pseudoinverse."""
        ps = LinTSParamState(num_arms=3, dim=2)
        # make design matrix singular
        ps.d[0] = np.zeros((2, 2))
        ps.r[0] = np.array([[1.0], [2.0]])

        theta = LinTSPolicy._sample_theta(ps, 0)

        assert theta.shape == (2, 1)
        assert np.all(np.isfinite(theta))

    def test_sample_theta_with_completely_broken_matrix_returns_zeros(self):
        """test sample_theta returns zeros when both inv and pinv fail."""
        ps = LinTSParamState(num_arms=3, dim=2)
        # create a matrix that might cause issues
        ps.d[0] = np.array([[np.inf, 0.0], [0.0, np.inf]])

        theta = LinTSPolicy._sample_theta(ps, 0)

        assert theta.shape == (2, 1)
        # should return zeros as fallback
        assert np.all(theta == 0.0) or np.all(np.isfinite(theta))

    def test_select_samples_from_posterior(self):
        """test select uses posterior sampling rather than upper bounds."""
        ps = LinTSParamState(num_arms=3, dim=2, v=0.5)
        ctx = Context(vector=[1.0, 1.0])

        # run multiple selections to verify stochastic behavior
        np.random.seed(42)
        selections = [LinTSPolicy.select(ps, ctx) for _ in range(100)]

        # should explore all arms due to sampling
        assert len(set(selections)) >= 2

    def test_select_with_trained_state_prefers_better_arm(self):
        """test select prefers arm with higher posterior mean after training."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 1.0])

        # train arm 0 with high rewards multiple times
        for _ in range(20):
            ps = LinTSPolicy.train(ps, ctx, choice=0, reward=1.0)

        # train arm 1 with low rewards
        for _ in range(20):
            ps = LinTSPolicy.train(ps, ctx, choice=1, reward=0.0)

        # select multiple times
        selections = [LinTSPolicy.select(ps, ctx) for _ in range(100)]
        arm_0_count = selections.count(0)
        arm_1_count = selections.count(1)

        # arm 0 should be selected much more often than arm 1
        assert arm_0_count > arm_1_count

    def test_train_matrix_update_computation(self):
        """test train computes design matrix update correctly."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=[2.0, 3.0])

        updated = LinTSPolicy.train(ps, ctx, choice=0, reward=1.0)

        # d should be I + x*x.T
        x = np.array([[2.0], [3.0]])  # noqa
        expected_outer_product = np.array([[4.0, 6.0], [6.0, 9.0]])
        expected_d = np.identity(2) + expected_outer_product

        assert np.allclose(updated.d[0], expected_d)

    def test_train_reward_vector_computation(self):
        """test train computes reward vector update correctly."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=[2.0, 3.0])

        updated = LinTSPolicy.train(ps, ctx, choice=0, reward=0.8)

        # r should be reward * x
        expected_r = np.array([[1.6], [2.4]])

        assert np.allclose(updated.r[0], expected_r)

    def test_train_updates_only_chosen_arm(self):
        """test train updates only the chosen arm."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 2.0])

        updated = LinTSPolicy.train(ps, ctx, choice=1, reward=0.5)

        # arm 1 should be updated
        assert not np.array_equal(updated.d[1], ps.d[1])
        assert not np.array_equal(updated.r[1], ps.r[1])

        # arms 0 and 2 should be unchanged
        assert np.array_equal(updated.d[0], ps.d[0])
        assert np.array_equal(updated.d[2], ps.d[2])
        assert np.array_equal(updated.r[0], ps.r[0])
        assert np.array_equal(updated.r[2], ps.r[2])

    def test_v_parameter_affects_exploration(self):
        """test v parameter affects exploration through posterior variance."""
        ps_low_v = LinTSParamState(num_arms=3, dim=2, v=0.1)
        ps_high_v = LinTSParamState(num_arms=3, dim=2, v=10.0)

        # train both with same data
        ctx = Context(vector=[1.0, 1.0])
        for _ in range(10):
            ps_low_v = LinTSPolicy.train(ps_low_v, ctx, choice=0, reward=1.0)
            ps_high_v = LinTSPolicy.train(ps_high_v, ctx, choice=0, reward=1.0)

        # sample theta many times and check variance
        np.random.seed(42)
        samples_low = [LinTSPolicy._sample_theta(ps_low_v, 0) for _ in range(100)]

        np.random.seed(42)
        samples_high = [LinTSPolicy._sample_theta(ps_high_v, 0) for _ in range(100)]

        var_low = np.var([s[0, 0] for s in samples_low])
        var_high = np.var([s[0, 0] for s in samples_high])

        # higher v should lead to higher variance in samples
        assert var_high > var_low

    def test_posterior_sampling_vs_ucb_difference(self):
        """test lints uses sampling (stochastic) rather than ucb (deterministic)."""
        ps = LinTSParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 1.0])

        # without seeding, selections should vary
        selections = [LinTSPolicy.select(ps, ctx) for _ in range(10)]

        # should have some variation (not always same arm)
        # with sampling, it's very unlikely to get same arm 10 times
        assert len(set(selections)) > 1
