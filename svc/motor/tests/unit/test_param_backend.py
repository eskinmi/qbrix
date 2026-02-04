import pytest
import numpy as np

from qbrixcore.policy.stochastic.ts import BetaTSPolicy
from qbrixcore.policy.stochastic.ts import GaussianTSPolicy

from motorsvc.param_backend import RedisBackedInMemoryParamBackend


class TestRedisBackedInMemoryParamBackend:

    def test_get_returns_cached_params(
        self, tenant_id, mock_redis_client, motor_cache, beta_ts_params
    ):
        backend = RedisBackedInMemoryParamBackend(mock_redis_client, motor_cache)
        experiment_id = "exp-123"

        motor_cache.set_params(tenant_id, experiment_id, beta_ts_params)
        result = backend.get(tenant_id, experiment_id)

        assert result is not None
        assert result.num_arms == 3
        assert np.array_equal(result.alpha, beta_ts_params.alpha)

    def test_get_returns_none_when_not_cached(self, tenant_id, mock_redis_client, motor_cache):
        backend = RedisBackedInMemoryParamBackend(mock_redis_client, motor_cache)

        result = backend.get(tenant_id, "nonexistent-exp")

        assert result is None

    def test_set_caches_params(
        self, tenant_id, mock_redis_client, motor_cache, beta_ts_params
    ):
        backend = RedisBackedInMemoryParamBackend(mock_redis_client, motor_cache)
        experiment_id = "exp-123"

        backend.set(tenant_id, experiment_id, beta_ts_params)

        cached = motor_cache.get_params(tenant_id, experiment_id)
        assert cached is not None
        assert np.array_equal(cached.alpha, beta_ts_params.alpha)

    @pytest.mark.asyncio
    async def test_update_params_fetches_from_redis_and_caches(
        self, tenant_id, mock_redis_client, motor_cache
    ):
        backend = RedisBackedInMemoryParamBackend(mock_redis_client, motor_cache)
        experiment_id = "exp-123"

        # mock redis returning params dict
        params_dict = {
            "num_arms": 3,
            "alpha_prior": 1.0,
            "beta_prior": 1.0,
            "alpha": [2.0, 1.5, 1.0],
            "beta": [1.0, 1.5, 2.0],
            "T": [1, 1, 1],
        }
        mock_redis_client.get_params.return_value = params_dict

        result = await backend.update_params(tenant_id, experiment_id, BetaTSPolicy)

        assert result is not None
        assert result.num_arms == 3
        assert np.array_equal(result.alpha, np.array([2.0, 1.5, 1.0]))

        # verify it was cached
        cached = motor_cache.get_params(tenant_id, experiment_id)
        assert cached is not None
        assert np.array_equal(cached.alpha, result.alpha)

    @pytest.mark.asyncio
    async def test_update_params_returns_none_when_redis_has_no_params(
        self, tenant_id, mock_redis_client, motor_cache
    ):
        backend = RedisBackedInMemoryParamBackend(mock_redis_client, motor_cache)
        experiment_id = "exp-123"

        mock_redis_client.get_params.return_value = None

        result = await backend.update_params(tenant_id, experiment_id, BetaTSPolicy)

        assert result is None

    @pytest.mark.asyncio
    async def test_update_params_uses_correct_policy_param_state_class(
        self, tenant_id, mock_redis_client, motor_cache
    ):
        backend = RedisBackedInMemoryParamBackend(mock_redis_client, motor_cache)
        experiment_id = "exp-123"

        # gaussian ts params
        params_dict = {
            "num_arms": 3,
            "prior_mean": 0.0,
            "prior_precision": 1.0,
            "noise_precision": 1.0,
            "posterior_mean": [0.5, 0.3, 0.7],
            "posterior_precision": [2.0, 2.0, 2.0],
            "T": [5, 5, 5],
        }
        mock_redis_client.get_params.return_value = params_dict

        result = await backend.update_params(tenant_id, experiment_id, GaussianTSPolicy)

        assert result is not None
        assert isinstance(result, GaussianTSPolicy.param_state_cls)
        assert result.num_arms == 3
        assert np.array_equal(result.posterior_mean, np.array([0.5, 0.3, 0.7]))

    @pytest.mark.asyncio
    async def test_update_params_validates_params_dict(
        self, tenant_id, mock_redis_client, motor_cache
    ):
        backend = RedisBackedInMemoryParamBackend(mock_redis_client, motor_cache)
        experiment_id = "exp-123"

        # invalid params dict (missing required fields)
        # note: pydantic will use defaults if fields are missing, so this won't fail
        # , but we can test that the params are properly validated
        params_dict = {
            "num_arms": 3,
            "alpha_prior": 1.0,
            "beta_prior": 1.0,
        }
        mock_redis_client.get_params.return_value = params_dict

        result = await backend.update_params(tenant_id, experiment_id, BetaTSPolicy)

        # params should be created with defaults for missing fields
        assert result is not None
        assert result.num_arms == 3
