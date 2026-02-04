"""Unit tests for LinUCBPolicy."""

import numpy as np
import pytest

from qbrixcore.policy.contextual.ucb import LinUCBPolicy
from qbrixcore.policy.contextual.ucb import LinUCBParamState
from qbrixcore.context import Context


class TestLinUCBParamState:
    def test_param_state_creation_minimal(self):
        """test param state creation with minimal args."""
        ps = LinUCBParamState(num_arms=3, dim=5)

        assert ps.num_arms == 3
        assert ps.dim == 5
        assert ps.alpha == 1.5
        assert ps.d.shape == (3, 5, 5)
        assert ps.r.shape == (3, 5, 1)

    def test_param_state_default_design_matrix_is_identity(self):
        """test default design matrix is identity for each arm."""
        ps = LinUCBParamState(num_arms=3, dim=4)

        for i in range(3):
            assert np.array_equal(ps.d[i], np.identity(4))

    def test_param_state_default_reward_vector_is_zero(self):
        """test default reward vector is zero for each arm."""
        ps = LinUCBParamState(num_arms=3, dim=4)

        for i in range(3):
            assert np.all(ps.r[i] == 0.0)

    def test_param_state_creation_with_custom_alpha(self):
        """test param state creation with custom alpha."""
        ps = LinUCBParamState(num_arms=3, dim=5, alpha=2.0)

        assert ps.alpha == 2.0

    def test_param_state_creation_with_custom_matrices(self):
        """test param state creation with pre-initialized matrices."""
        d = np.array([np.identity(3) * 2 for _ in range(2)])
        r = np.ones((2, 3, 1))

        ps = LinUCBParamState(num_arms=2, dim=3, d=d, r=r)

        assert np.array_equal(ps.d, d)
        assert np.array_equal(ps.r, r)

    def test_param_state_validation_num_arms_positive(self):
        """test param state validation requires positive num_arms."""
        with pytest.raises(ValueError):
            LinUCBParamState(num_arms=0, dim=5)

    def test_param_state_validation_dim_positive(self):
        """test param state validation requires positive dim."""
        with pytest.raises(ValueError):
            LinUCBParamState(num_arms=3, dim=0)

    def test_param_state_validation_alpha_positive(self):
        """test param state validation requires positive alpha."""
        with pytest.raises(ValueError):
            LinUCBParamState(num_arms=3, dim=5, alpha=0.0)


class TestLinUCBPolicy:
    def test_policy_name(self):
        """test policy has correct name."""
        assert LinUCBPolicy.name == "LinUCBPolicy"

    def test_policy_param_state_cls(self):
        """test policy has correct param state class."""
        assert LinUCBPolicy.param_state_cls == LinUCBParamState

    def test_init_params(self):
        """test init_params creates correct param state."""
        params = LinUCBPolicy.init_params(num_arms=3, dim=5)

        assert isinstance(params, LinUCBParamState)
        assert params.num_arms == 3
        assert params.dim == 5

    def test_reshape_context_vector_from_list(self):
        """test reshaping context vector from list to column vector."""
        ctx = Context(vector=[1.0, 2.0, 3.0])

        reshaped = LinUCBPolicy._reshape_context_vector(ctx)

        assert isinstance(reshaped, np.ndarray)
        assert reshaped.shape == (3, 1)
        assert np.array_equal(reshaped, [[1.0], [2.0], [3.0]])

    def test_reshape_context_vector_from_1d_array(self):
        """test reshaping 1d numpy array to column vector."""
        ctx = Context(vector=np.array([1.0, 2.0, 3.0]))

        reshaped = LinUCBPolicy._reshape_context_vector(ctx)

        assert reshaped.shape == (3, 1)

    def test_reshape_context_vector_already_column(self):
        """test context vector already in column form is unchanged."""
        ctx = Context(vector=np.array([[1.0], [2.0], [3.0]]))

        reshaped = LinUCBPolicy._reshape_context_vector(ctx)

        assert reshaped.shape == (3, 1)

    def test_select_returns_valid_arm_index(self):
        """test select returns valid arm index."""
        ps = LinUCBParamState(num_arms=3, dim=4)
        ctx = Context(vector=[1.0, 0.5, -0.3, 0.8])

        arm_index = LinUCBPolicy.select(ps, ctx)

        assert isinstance(arm_index, int)
        assert 0 <= arm_index < 3

    def test_select_with_initial_state_all_arms_equal(self):
        """test select with initial state explores randomly."""
        ps = LinUCBParamState(num_arms=3, dim=4)
        ctx = Context(vector=[1.0, 0.5, -0.3, 0.8])

        # all arms should have equal upper bounds initially
        # so selection should be deterministic based on argmax
        arm_index = LinUCBPolicy.select(ps, ctx)

        assert arm_index in [0, 1, 2]

    def test_select_prefers_arm_with_higher_estimate(self):
        """test select prefers arm with higher reward estimate."""
        ps = LinUCBParamState(num_arms=3, dim=2)

        # manually set r to give arm 0 higher estimate
        ps.r[0] = np.array([[10.0], [10.0]])  # high reward history
        ps.r[1] = np.array([[1.0], [1.0]])
        ps.r[2] = np.array([[1.0], [1.0]])

        ctx = Context(vector=[1.0, 1.0])

        arm_index = LinUCBPolicy.select(ps, ctx)

        # arm 0 should be preferred
        assert arm_index == 0

    def test_train_updates_design_matrix(self):
        """test train updates design matrix for chosen arm."""
        ps = LinUCBParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 2.0])

        updated = LinUCBPolicy.train(ps, ctx, choice=1, reward=0.5)

        # d[1] should be updated with x*x.T
        expected_update = np.array([[1.0, 2.0], [2.0, 4.0]])
        expected_d = np.identity(2) + expected_update

        assert np.allclose(updated.d[1], expected_d)
        # other arms unchanged
        assert np.array_equal(updated.d[0], ps.d[0])
        assert np.array_equal(updated.d[2], ps.d[2])

    def test_train_updates_reward_vector(self):
        """test train updates reward vector for chosen arm."""
        ps = LinUCBParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 2.0])

        updated = LinUCBPolicy.train(ps, ctx, choice=1, reward=0.5)

        # r[1] should be updated with reward * x
        expected_r = np.array([[0.5], [1.0]])

        assert np.allclose(updated.r[1], expected_r)
        # other arms unchanged
        assert np.array_equal(updated.r[0], ps.r[0])
        assert np.array_equal(updated.r[2], ps.r[2])

    def test_train_does_not_mutate_original_params(self):
        """test train returns new params without mutating original."""
        ps = LinUCBParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 2.0])
        original_d = ps.d.copy()
        original_r = ps.r.copy()

        updated = LinUCBPolicy.train(ps, ctx, choice=1, reward=0.5)

        # original unchanged
        assert np.array_equal(ps.d, original_d)
        assert np.array_equal(ps.r, original_r)
        # updated changed
        assert not np.array_equal(updated.d[1], ps.d[1])

    def test_train_multiple_updates_same_arm(self):
        """test train accumulates updates for same arm."""
        ps = LinUCBParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 1.0])

        ps = LinUCBPolicy.train(ps, ctx, choice=0, reward=1.0)
        ps = LinUCBPolicy.train(ps, ctx, choice=0, reward=0.5)
        ps = LinUCBPolicy.train(ps, ctx, choice=0, reward=0.0)

        # design matrix should accumulate
        assert not np.array_equal(ps.d[0], np.identity(2))
        # reward vector should accumulate
        expected_r_sum = 1.0 + 0.5 + 0.0  # total reward
        assert ps.r[0][0, 0] == pytest.approx(expected_r_sum)

    def test_train_with_list_context_vector(self):
        """test train works with list context vector."""
        ps = LinUCBParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 2.0])

        updated = LinUCBPolicy.train(ps, ctx, choice=0, reward=0.5)

        assert updated is not None
        assert not np.array_equal(updated.d[0], ps.d[0])

    def test_train_with_numpy_context_vector(self):
        """test train works with numpy array context vector."""
        ps = LinUCBParamState(num_arms=3, dim=2)
        ctx = Context(vector=np.array([1.0, 2.0]))

        updated = LinUCBPolicy.train(ps, ctx, choice=0, reward=0.5)

        assert updated is not None
        assert not np.array_equal(updated.d[0], ps.d[0])

    def test_train_with_negative_reward(self):
        """test train handles negative rewards."""
        ps = LinUCBParamState(num_arms=3, dim=2)
        ctx = Context(vector=[1.0, 1.0])

        updated = LinUCBPolicy.train(ps, ctx, choice=0, reward=-1.0)

        assert updated.r[0][0, 0] == -1.0
        assert updated.r[0][1, 0] == -1.0

    def test_arm_upper_bound_calculation(self):
        """test upper bound calculation for an arm."""
        ps = LinUCBParamState(num_arms=3, dim=2)
        ctx_vector = np.array([[1.0], [1.0]])

        upper_bound = LinUCBPolicy._arm_upper_bound(ps, 0, ctx_vector)

        assert isinstance(upper_bound, float)
        assert upper_bound >= 0.0  # with initial state, should be positive

    def test_arm_upper_bound_with_singular_matrix(self):
        """test upper bound returns inf for singular matrix."""
        ps = LinUCBParamState(num_arms=3, dim=2)
        # make design matrix singular
        ps.d[0] = np.zeros((2, 2))
        ctx_vector = np.array([[1.0], [1.0]])

        upper_bound = LinUCBPolicy._arm_upper_bound(ps, 0, ctx_vector)

        assert upper_bound == float("inf")

    def test_select_single_arm(self):
        """test select with single arm always returns 0."""
        ps = LinUCBParamState(num_arms=1, dim=3)
        ctx = Context(vector=[1.0, 0.5, -0.3])

        arm_index = LinUCBPolicy.select(ps, ctx)

        assert arm_index == 0

    def test_integration_select_and_train(self):
        """test integration of select and train workflow."""
        ps = LinUCBParamState(num_arms=3, dim=4)
        ctx = Context(vector=[1.0, 0.5, -0.3, 0.8])

        # select arm
        arm = LinUCBPolicy.select(ps, ctx)
        assert 0 <= arm < 3

        # train on selection
        updated_ps = LinUCBPolicy.train(ps, ctx, choice=arm, reward=1.0)

        # select again with updated params
        arm2 = LinUCBPolicy.select(updated_ps, ctx)
        assert 0 <= arm2 < 3
