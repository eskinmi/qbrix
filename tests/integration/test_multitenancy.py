"""integration tests for multi-tenancy isolation."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from qbrixstore.postgres.models import Tenant, Pool, Experiment
from qbrixstore.redis.client import RedisClient
from qbrixstore.redis.streams import FeedbackEvent

from proxysvc.repository import (
    TenantRepository,
    PoolRepository,
    ExperimentRepository,
)


class TestTenantRepository:
    """tests for tenant repository."""

    @pytest.mark.asyncio
    async def test_create_tenant(self, db_session: AsyncSession):
        repo = TenantRepository(db_session)
        tenant = await repo.create(name="Test Tenant", slug="test-tenant")

        assert tenant.id is not None
        assert tenant.name == "Test Tenant"
        assert tenant.slug == "test-tenant"
        assert tenant.is_active is True

    @pytest.mark.asyncio
    async def test_get_tenant_by_slug(self, db_session: AsyncSession, tenant_a: Tenant):
        repo = TenantRepository(db_session)
        tenant = await repo.get_by_slug("tenant-a")

        assert tenant is not None
        assert tenant.id == "tenant-a"

    @pytest.mark.asyncio
    async def test_list_tenants(self, db_session: AsyncSession, tenant_a: Tenant, tenant_b: Tenant):
        repo = TenantRepository(db_session)
        tenants = await repo.list()

        assert len(tenants) == 2


class TestPoolTenantIsolation:
    """tests for pool tenant isolation."""

    @pytest.mark.asyncio
    async def test_pool_created_in_tenant(self, db_session: AsyncSession, tenant_a: Tenant):
        repo = PoolRepository(db_session, tenant_a.id)
        pool = await repo.create(name="test-pool", arms=[{"name": "arm-0"}])

        assert pool.tenant_id == tenant_a.id

    @pytest.mark.asyncio
    async def test_same_pool_name_different_tenants(
        self, db_session: AsyncSession, tenant_a: Tenant, tenant_b: Tenant
    ):
        """same pool name should be allowed in different tenants."""
        repo_a = PoolRepository(db_session, tenant_a.id)
        repo_b = PoolRepository(db_session, tenant_b.id)

        pool_a = await repo_a.create(name="shared-name", arms=[{"name": "arm-0"}])
        pool_b = await repo_b.create(name="shared-name", arms=[{"name": "arm-0"}])

        assert pool_a.name == pool_b.name
        assert pool_a.id != pool_b.id
        assert pool_a.tenant_id != pool_b.tenant_id

    @pytest.mark.asyncio
    async def test_pool_list_filtered_by_tenant(
        self, db_session: AsyncSession, tenant_a: Tenant, tenant_b: Tenant
    ):
        """pool list should only return pools for the given tenant."""
        repo_a = PoolRepository(db_session, tenant_a.id)
        repo_b = PoolRepository(db_session, tenant_b.id)

        await repo_a.create(name="pool-a1", arms=[{"name": "arm-0"}])
        await repo_a.create(name="pool-a2", arms=[{"name": "arm-0"}])
        await repo_b.create(name="pool-b1", arms=[{"name": "arm-0"}])

        pools_a = await repo_a.list()
        pools_b = await repo_b.list()

        assert len(pools_a) == 2
        assert len(pools_b) == 1
        assert all(p.tenant_id == tenant_a.id for p in pools_a)
        assert all(p.tenant_id == tenant_b.id for p in pools_b)

    @pytest.mark.asyncio
    async def test_pool_get_requires_tenant_match(
        self, db_session: AsyncSession, tenant_a: Tenant, tenant_b: Tenant
    ):
        """pool get should not return pools from other tenants."""
        repo_a = PoolRepository(db_session, tenant_a.id)
        repo_b = PoolRepository(db_session, tenant_b.id)

        pool_a = await repo_a.create(name="pool-a", arms=[{"name": "arm-0"}])

        # tenant_b should not be able to access tenant_a's pool
        result = await repo_b.get(pool_a.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_pool_delete_requires_tenant_match(
        self, db_session: AsyncSession, tenant_a: Tenant, tenant_b: Tenant
    ):
        """pool delete should not delete pools from other tenants."""
        repo_a = PoolRepository(db_session, tenant_a.id)
        repo_b = PoolRepository(db_session, tenant_b.id)

        pool_a = await repo_a.create(name="pool-a", arms=[{"name": "arm-0"}])

        # tenant_b should not be able to delete tenant_a's pool
        deleted = await repo_b.delete(pool_a.id)
        assert deleted is False

        # pool should still exist
        result = await repo_a.get(pool_a.id)
        assert result is not None


class TestExperimentTenantIsolation:
    """tests for experiment tenant isolation."""

    @pytest.mark.asyncio
    async def test_experiment_created_in_tenant(
        self, db_session: AsyncSession, tenant_a: Tenant
    ):
        pool_repo = PoolRepository(db_session, tenant_a.id)
        pool = await pool_repo.create(name="test-pool", arms=[{"name": "arm-0"}])

        exp_repo = ExperimentRepository(db_session, tenant_a.id)
        experiment = await exp_repo.create(
            name="test-exp",
            pool_id=pool.id,
            protocol="beta_ts",
            protocol_params={},
            enabled=True,
        )

        assert experiment.tenant_id == tenant_a.id

    @pytest.mark.asyncio
    async def test_same_experiment_name_different_tenants(
        self, db_session: AsyncSession, tenant_a: Tenant, tenant_b: Tenant
    ):
        """same experiment name should be allowed in different tenants."""
        pool_repo_a = PoolRepository(db_session, tenant_a.id)
        pool_repo_b = PoolRepository(db_session, tenant_b.id)
        pool_a = await pool_repo_a.create(name="pool", arms=[{"name": "arm-0"}])
        pool_b = await pool_repo_b.create(name="pool", arms=[{"name": "arm-0"}])

        exp_repo_a = ExperimentRepository(db_session, tenant_a.id)
        exp_repo_b = ExperimentRepository(db_session, tenant_b.id)

        exp_a = await exp_repo_a.create(
            name="shared-name",
            pool_id=pool_a.id,
            protocol="beta_ts",
            protocol_params={},
            enabled=True,
        )
        exp_b = await exp_repo_b.create(
            name="shared-name",
            pool_id=pool_b.id,
            protocol="beta_ts",
            protocol_params={},
            enabled=True,
        )

        assert exp_a.name == exp_b.name
        assert exp_a.id != exp_b.id
        assert exp_a.tenant_id != exp_b.tenant_id

    @pytest.mark.asyncio
    async def test_experiment_list_filtered_by_tenant(
        self, db_session: AsyncSession, tenant_a: Tenant, tenant_b: Tenant
    ):
        """experiment list should only return experiments for the given tenant."""
        pool_repo_a = PoolRepository(db_session, tenant_a.id)
        pool_repo_b = PoolRepository(db_session, tenant_b.id)
        pool_a = await pool_repo_a.create(name="pool", arms=[{"name": "arm-0"}])
        pool_b = await pool_repo_b.create(name="pool", arms=[{"name": "arm-0"}])

        exp_repo_a = ExperimentRepository(db_session, tenant_a.id)
        exp_repo_b = ExperimentRepository(db_session, tenant_b.id)

        await exp_repo_a.create(
            name="exp-a1", pool_id=pool_a.id, protocol="beta_ts", protocol_params={}, enabled=True
        )
        await exp_repo_a.create(
            name="exp-a2", pool_id=pool_a.id, protocol="beta_ts", protocol_params={}, enabled=True
        )
        await exp_repo_b.create(
            name="exp-b1", pool_id=pool_b.id, protocol="beta_ts", protocol_params={}, enabled=True
        )

        exps_a = await exp_repo_a.list()
        exps_b = await exp_repo_b.list()

        assert len(exps_a) == 2
        assert len(exps_b) == 1
        assert all(e.tenant_id == tenant_a.id for e in exps_a)
        assert all(e.tenant_id == tenant_b.id for e in exps_b)

    @pytest.mark.asyncio
    async def test_experiment_get_requires_tenant_match(
        self, db_session: AsyncSession, tenant_a: Tenant, tenant_b: Tenant
    ):
        """experiment get should not return experiments from other tenants."""
        pool_repo_a = PoolRepository(db_session, tenant_a.id)
        pool_a = await pool_repo_a.create(name="pool", arms=[{"name": "arm-0"}])

        exp_repo_a = ExperimentRepository(db_session, tenant_a.id)
        exp_repo_b = ExperimentRepository(db_session, tenant_b.id)

        exp_a = await exp_repo_a.create(
            name="exp-a", pool_id=pool_a.id, protocol="beta_ts", protocol_params={}, enabled=True
        )

        # tenant_b should not be able to access tenant_a's experiment
        result = await exp_repo_b.get(exp_a.id)
        assert result is None


class TestRedisKeyIsolation:
    """tests for redis key isolation patterns."""

    def test_param_key_includes_tenant(self):
        """param key should include tenant_id."""
        key = RedisClient._param_key("tenant-123", "exp-456")
        assert key == "qbrix:tenant:tenant-123:params:exp-456"
        assert "tenant-123" in key

    def test_experiment_key_includes_tenant(self):
        """experiment key should include tenant_id."""
        key = RedisClient._experiment_key("tenant-123", "exp-456")
        assert key == "qbrix:tenant:tenant-123:experiment:exp-456"
        assert "tenant-123" in key

    def test_gate_key_includes_tenant(self):
        """gate key should include tenant_id."""
        key = RedisClient._gate_key("tenant-123", "exp-456")
        assert key == "qbrix:tenant:tenant-123:gate:exp-456"
        assert "tenant-123" in key

    def test_different_tenants_different_keys(self):
        """different tenants should have different keys for same experiment."""
        key_a = RedisClient._param_key("tenant-a", "exp-123")
        key_b = RedisClient._param_key("tenant-b", "exp-123")

        assert key_a != key_b
        assert "tenant-a" in key_a
        assert "tenant-b" in key_b


class TestFeedbackEventTenantIsolation:
    """tests for feedback event tenant isolation."""

    def test_feedback_event_has_tenant_id(self):
        """feedback event should include tenant_id."""
        event = FeedbackEvent(
            tenant_id="tenant-123",
            experiment_id="exp-456",
            request_id="req-789",
            arm_index=0,
            reward=1.0,
            context_id="ctx-abc",
            context_vector=[0.1, 0.2],
            context_metadata={"key": "value"},
            timestamp_ms=1234567890000,
        )

        assert event.tenant_id == "tenant-123"

    def test_feedback_event_serialization_includes_tenant(self):
        """feedback event to_dict should include tenant_id."""
        event = FeedbackEvent(
            tenant_id="tenant-123",
            experiment_id="exp-456",
            request_id="req-789",
            arm_index=0,
            reward=1.0,
            context_id="ctx-abc",
            context_vector=[0.1, 0.2],
            context_metadata={},
            timestamp_ms=1234567890000,
        )

        data = event.to_dict()
        assert "tenant_id" in data
        assert data["tenant_id"] == "tenant-123"

    def test_feedback_event_grouping_by_tenant_experiment(self):
        """feedback events should be groupable by (tenant_id, experiment_id)."""
        events = [
            FeedbackEvent(
                tenant_id="tenant-a", experiment_id="exp-1", request_id="r1",
                arm_index=0, reward=1.0, context_id="c1", context_vector=[],
                context_metadata={}, timestamp_ms=1000
            ),
            FeedbackEvent(
                tenant_id="tenant-a", experiment_id="exp-1", request_id="r2",
                arm_index=1, reward=0.5, context_id="c2", context_vector=[],
                context_metadata={}, timestamp_ms=2000
            ),
            FeedbackEvent(
                tenant_id="tenant-b", experiment_id="exp-1", request_id="r3",
                arm_index=0, reward=0.8, context_id="c3", context_vector=[],
                context_metadata={}, timestamp_ms=3000
            ),
        ]

        # group by (tenant_id, experiment_id)
        from collections import defaultdict
        grouped = defaultdict(list)
        for event in events:
            key = (event.tenant_id, event.experiment_id)
            grouped[key].append(event)

        # tenant-a:exp-1 has 2 events, tenant-b:exp-1 has 1 event
        assert len(grouped[("tenant-a", "exp-1")]) == 2
        assert len(grouped[("tenant-b", "exp-1")]) == 1
