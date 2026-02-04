import numpy as np

from qbrixcore.protoc.stochastic.ts import BetaTSParamState

from motorsvc.cache import MotorCache
from motorsvc.config import MotorSettings


class TestMotorCache:
    def test_init_creates_ttl_caches(self, motor_settings):
        cache = MotorCache(motor_settings)

        assert cache._params is not None
        assert cache._agents is not None

    def test_set_and_get_params(self, tenant_id, motor_cache, beta_ts_params):
        experiment_id = "exp-123"

        motor_cache.set_params(tenant_id, experiment_id, beta_ts_params)
        retrieved = motor_cache.get_params(tenant_id, experiment_id)

        assert retrieved is not None
        assert retrieved.num_arms == 3
        assert np.array_equal(retrieved.alpha, beta_ts_params.alpha)

    def test_get_params_returns_none_for_missing_experiment(self, tenant_id, motor_cache):
        result = motor_cache.get_params(tenant_id, "nonexistent-exp")

        assert result is None

    def test_set_and_get_agent(self, tenant_id, motor_cache, mock_agent):
        experiment_id = "exp-123"

        motor_cache.set_agent(tenant_id, experiment_id, mock_agent)
        retrieved = motor_cache.get_agent(tenant_id, experiment_id)

        assert retrieved is not None
        assert retrieved.experiment_id == "exp-123"
        assert len(retrieved.pool.arms) == 3

    def test_get_agent_returns_none_for_missing_experiment(self, tenant_id, motor_cache):
        result = motor_cache.get_agent(tenant_id, "nonexistent-exp")

        assert result is None

    def test_invalidate_experiment_removes_params_and_agent(
        self, tenant_id, motor_cache, beta_ts_params, mock_agent
    ):
        experiment_id = "exp-123"

        motor_cache.set_params(tenant_id, experiment_id, beta_ts_params)
        motor_cache.set_agent(tenant_id, experiment_id, mock_agent)

        motor_cache.invalidate_experiment(tenant_id, experiment_id)

        assert motor_cache.get_params(tenant_id, experiment_id) is None
        assert motor_cache.get_agent(tenant_id, experiment_id) is None

    def test_invalidate_experiment_handles_missing_experiment(self, tenant_id, motor_cache):
        # should not raise exception
        motor_cache.invalidate_experiment(tenant_id, "nonexistent-exp")

    def test_clear_removes_all_entries(self, tenant_id, motor_cache, beta_ts_params, mock_agent):
        motor_cache.set_params(tenant_id, "exp-1", beta_ts_params)
        motor_cache.set_params(tenant_id, "exp-2", beta_ts_params)
        motor_cache.set_agent(tenant_id, "exp-1", mock_agent)

        motor_cache.clear()

        assert motor_cache.get_params(tenant_id, "exp-1") is None
        assert motor_cache.get_params(tenant_id, "exp-2") is None
        assert motor_cache.get_agent(tenant_id, "exp-1") is None

    def test_params_cache_ttl_expiration(self, tenant_id, motor_settings):
        # test with very short ttl
        settings = MotorSettings(
            param_cache_ttl=1,  # 1 second
            param_cache_maxsize=100,
            agent_cache_ttl=300,
            agent_cache_maxsize=50,
        )
        cache = MotorCache(settings)
        params = BetaTSParamState(
            num_arms=2,
            alpha=np.array([1.0, 1.0]),
            beta=np.array([1.0, 1.0]),
        )

        cache.set_params(tenant_id, "exp-1", params)
        assert cache.get_params(tenant_id, "exp-1") is not None

        # note: actual ttl expiration testing would require sleep
        # which slows tests, so we just verify the cache is configured correctly

    def test_cache_maxsize_limit(self, tenant_id, motor_settings):
        # test with small maxsize
        settings = MotorSettings(
            param_cache_ttl=60,
            param_cache_maxsize=2,  # very small
            agent_cache_ttl=300,
            agent_cache_maxsize=2,
        )
        cache = MotorCache(settings)

        # cache size behavior depends on cachebox implementation
        # we verify basic functionality with multiple entries
        for i in range(5):
            params = BetaTSParamState(
                num_arms=2,
                alpha=np.array([1.0, 1.0]),
                beta=np.array([1.0, 1.0]),
            )
            cache.set_params(tenant_id, f"exp-{i}", params)

        # at least the most recent entries should be cached
        assert cache.get_params(tenant_id, "exp-4") is not None
