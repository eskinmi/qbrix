from __future__ import annotations

from zoneinfo import ZoneInfo
from datetime import datetime
from datetime import timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from qbrixstore.postgres.models import Tenant, Pool, Arm, Experiment, FeatureGate
from qbrixstore.postgres.models import User
from qbrixstore.postgres.models import APIKey

from proxysvc.gate.config import FeatureGateConfig
from proxysvc.gate.model.base import BaseArmModel, ArmConfig
from proxysvc.gate.model.experiment import (
    ExperimentConfig,
    RolloutConfig,
    ScheduleConfig,
    ActiveHoursConfig,
    ActivePeriodConfig,
)
from proxysvc.gate.model.rule import Rule



class TenantRepository:

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, name: str, slug: str) -> Tenant:
        """
        Create a tenant for the given name and slug in the database.
        Parameters
        ----------
        name: string
            tenant name
        slug: string
            tenant slug

        Returns
        -------
            Tenant object
        """
        tenant = Tenant(name=name, slug=slug)
        self._session.add(tenant)
        await self._session.flush()
        return tenant

    async def get(self, tenant_id: str) -> Tenant | None:
        """
        Get a tenant for the given tenant_id from the database.
        Parameters
        ----------
        tenant_id: string

        Returns
        -------
        Tenant object
        """
        stmt = select(Tenant).where(Tenant.id == tenant_id)
        response = await self._session.execute(stmt)
        return response.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Tenant | None:
        """
        Get a tenant for the given slug from the database.

        Parameters
        ----------
        slug: string

        Returns
        -------
        Tenant object
        """
        stmt = select(Tenant).where(Tenant.slug == slug)
        response = await self._session.execute(stmt)
        return response.scalar_one_or_none()

    async def list(self, limit: int = 100, offset: int = 0) -> list[Tenant]:
        """
        List all the tenants in the database.
        Parameters
        ----------
        limit: int
            default: 1000
        offset: int
            default: 0

        Returns
        -------
        List of Tenant objects
        """
        stmt = (
            select(Tenant)
            .where(Tenant.is_active == True)
            .order_by(Tenant.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        response = await self._session.execute(stmt)
        return list(response.scalars().all())

    async def deactivate(self, tenant_id: str) -> bool:
        """
        Deactivate a tenant for the given tenant_id from the database.

        Parameters
        ----------
        tenant_id: string

        Returns
        -------
        Bool
        """
        tenant = await self.get(tenant_id)
        if tenant is None:
            return False
        tenant.is_active = False
        await self._session.flush()
        return True


class PoolRepository:

    def __init__(self, session: AsyncSession, tenant_id: str):
        self._session = session
        self._tenant_id = tenant_id

    async def create(self, name: str, arms: list[dict]) -> Pool:
        pool = Pool(name=name, tenant_id=self._tenant_id)
        for i, arm_data in enumerate(arms):
            arm = Arm(
                name=arm_data["name"], index=i, metadata_=arm_data.get("metadata", {})
            )
            pool.arms.append(arm)
        self._session.add(pool)
        await self._session.flush()
        return pool

    async def get(self, pool_id: str) -> Pool | None:
        stmt = (
            select(Pool)
            .options(selectinload(Pool.arms))
            .where(Pool.id == pool_id, Pool.tenant_id == self._tenant_id)
        )
        response = await self._session.execute(stmt)
        return response.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Pool | None:
        stmt = (
            select(Pool)
            .options(selectinload(Pool.arms))
            .where(Pool.name == name, Pool.tenant_id == self._tenant_id)
        )
        response = await self._session.execute(stmt)
        return response.scalar_one_or_none()

    async def delete(self, pool_id: str) -> bool:
        pool = await self.get(pool_id)
        if pool is None:
            return False
        await self._session.delete(pool)
        await self._session.flush()
        return True

    async def list(self, limit: int = 100, offset: int = 0) -> list[Pool]:
        stmt = (
            select(Pool)
            .options(selectinload(Pool.arms))
            .where(Pool.tenant_id == self._tenant_id)
            .order_by(Pool.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        response = await self._session.execute(stmt)
        return list(response.scalars().all())


class ExperimentRepository:

    def __init__(self, session: AsyncSession, tenant_id: str):
        self._session = session
        self._tenant_id = tenant_id

    async def create(
        self,
        name: str,
        pool_id: str,
        policy: str,
        policy_params: dict,
        enabled: bool,
        feature_gate_config: dict | None = None,
    ) -> Experiment:
        experiment = Experiment(
            name=name,
            tenant_id=self._tenant_id,
            pool_id=pool_id,
            policy=policy,
            policy_params=policy_params,
            enabled=enabled,
        )

        if feature_gate_config:
            feature_gate = FeatureGate(
                enabled=feature_gate_config.get("enabled", True),
                rollout_percentage=feature_gate_config.get("rollout_percentage", 100.0),
                default_arm_id=feature_gate_config.get("default_arm_id"),
                schedule_start=feature_gate_config.get("schedule_start"),
                schedule_end=feature_gate_config.get("schedule_end"),
                active_hours_start=feature_gate_config.get("active_hours_start"),
                active_hours_end=feature_gate_config.get("active_hours_end"),
                timezone=feature_gate_config.get("timezone", "UTC"),
                rules=feature_gate_config.get("rules", []),
                version=1,
            )
            experiment.feature_gate = feature_gate

        self._session.add(experiment)
        await self._session.flush()
        return experiment

    async def get(self, experiment_id: str) -> Experiment | None:
        stmt = (
            select(Experiment)
            .options(
                selectinload(Experiment.pool).selectinload(Pool.arms),
                selectinload(Experiment.feature_gate),
            )
            .where(Experiment.id == experiment_id, Experiment.tenant_id == self._tenant_id)
        )
        response = await self._session.execute(stmt)
        return response.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Experiment | None:
        stmt = (
            select(Experiment)
            .options(
                selectinload(Experiment.pool).selectinload(Pool.arms),
                selectinload(Experiment.feature_gate),
            )
            .where(Experiment.name == name, Experiment.tenant_id == self._tenant_id)
        )
        response = await self._session.execute(stmt)
        return response.scalar_one_or_none()

    async def update(self, experiment_id: str, **kwargs) -> Experiment | None:
        experiment = await self.get(experiment_id)
        if experiment is None:
            return None

        for key, value in kwargs.items():
            if hasattr(experiment, key) and value is not None:
                setattr(experiment, key, value)

        await self._session.flush()
        return experiment

    async def delete(self, experiment_id: str) -> bool:
        experiment = await self.get(experiment_id)
        if experiment is None:
            return False
        await self._session.delete(experiment)
        await self._session.flush()
        return True

    async def list(self, limit: int = 100, offset: int = 0) -> list[Experiment]:
        stmt = (
            select(Experiment)
            .options(
                selectinload(Experiment.pool).selectinload(Pool.arms),
                selectinload(Experiment.feature_gate),
            )
            .where(Experiment.tenant_id == self._tenant_id)
            .order_by(Experiment.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        response = await self._session.execute(stmt)
        return list(response.scalars().all())


class FeatureGateRepository:
    """repository for feature gate CRUD and transformation to pydantic config."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get(self, experiment_id: str) -> FeatureGate | None:
        """get feature gate by experiment id with default arm loaded."""
        stmt = (
            select(FeatureGate)
            .options(selectinload(FeatureGate.default_arm))
            .where(FeatureGate.experiment_id == experiment_id)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, experiment_id: str, config: dict) -> FeatureGate:
        """create a new feature gate from config dict."""
        gate = FeatureGate(
            experiment_id=experiment_id,
            enabled=config.get("enabled", True),
            rollout_percentage=config.get("rollout_percentage", 100.0),
            default_arm_id=config.get("default_arm_id"),
            schedule_start=config.get("schedule_start"),
            schedule_end=config.get("schedule_end"),
            active_hours_start=config.get("active_hours_start"),
            active_hours_end=config.get("active_hours_end"),
            timezone=config.get("timezone", "UTC"),
            rules=config.get("rules", []),
            version=1,
        )
        self._session.add(gate)
        await self._session.flush()
        return gate

    async def update(self, experiment_id: str, config: dict) -> FeatureGate | None:
        """update feature gate fields, incrementing version."""
        gate = await self.get(experiment_id)
        if gate is None:
            return None

        for key, value in config.items():
            if hasattr(gate, key) and value is not None:
                setattr(gate, key, value)

        gate.version += 1
        await self._session.flush()
        return gate

    async def delete(self, experiment_id: str) -> bool:
        """delete feature gate by experiment id."""
        gate = await self.get(experiment_id)
        if gate is None:
            return False
        await self._session.delete(gate)
        await self._session.flush()
        return True

    @staticmethod
    def to_config(gate: FeatureGate) -> FeatureGateConfig:
        """transform postgres feature gate to pydantic config."""
        tz = ZoneInfo(gate.timezone) if gate.timezone else ZoneInfo("UTC")

        committed_arm = BaseArmModel()
        if gate.default_arm:
            committed_arm = BaseArmModel(
                id=gate.default_arm.id,
                name=gate.default_arm.name,
                index=gate.default_arm.index,
            )

        experiment = ExperimentConfig(
            experiment_id=gate.experiment_id,
            enabled=gate.enabled,
            arm=ArmConfig(committed=committed_arm),
            rollout=RolloutConfig(percentage=gate.rollout_percentage),
            schedule=ScheduleConfig(
                hour=ActiveHoursConfig(
                    start=gate.active_hours_start,
                    end=gate.active_hours_end,
                    timezone=tz,
                ),
                period=ActivePeriodConfig(
                    start=gate.schedule_start, end=gate.schedule_end, timezone=tz
                ),
            ),
            updated_at=gate.updated_at,
            version=gate.version,
        )

        rules = [Rule.model_validate(r) for r in (gate.rules or [])]

        return FeatureGateConfig(
            experiment=experiment,
            rules=rules,
            updated_at=gate.updated_at,
            version=gate.version,
        )


class UserRepository:

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(
        self,
        email: str,
        password_hash: str,
        tenant_id: str,
        plan_tier: str = "free",
        role: str = "member",
    ) -> User:
        user = User(
            email=email,
            password_hash=password_hash,
            tenant_id=tenant_id,
            plan_tier=plan_tier,
            role=role,
        )
        self._session.add(user)
        await self._session.flush()
        return user

    async def get(self, user_id: str) -> User | None:
        stmt = (
            select(User).options(selectinload(User.api_keys)).where(User.id == user_id)
        )
        response = await self._session.execute(stmt)
        return response.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        stmt = (
            select(User).options(selectinload(User.api_keys)).where(User.email == email)
        )
        response = await self._session.execute(stmt)
        return response.scalar_one_or_none()

    async def update(self, user_id: str, **kwargs) -> User | None:
        user = await self.get(user_id)
        if user is None:
            return None

        for key, value in kwargs.items():
            if hasattr(user, key) and value is not None:
                setattr(user, key, value)

        await self._session.flush()
        return user

    async def deactivate(self, user_id: str) -> bool:
        user = await self.get(user_id)
        if user is None:
            return False
        user.is_active = False
        await self._session.flush()
        return True

    async def delete(self, user_id: str) -> bool:
        user = await self.get(user_id)
        if user is None:
            return False
        await self._session.delete(user)
        await self._session.flush()
        return True

    async def list(self, limit: int = 100, offset: int = 0) -> list[User]:
        stmt = (
            select(User)
            .options(selectinload(User.api_keys))
            .order_by(User.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        response = await self._session.execute(stmt)
        return list(response.scalars().all())


class APIKeyRepository:

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(
        self,
        user_id: str,
        key_hash: str,
        name: str = "Default API Key",
        rate_limit_per_minute: int = 1000,
        scopes: list[str] | None = None,
    ) -> APIKey:
        api_key = APIKey(
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            rate_limit_per_minute=rate_limit_per_minute,
            scopes=scopes or [],
        )
        self._session.add(api_key)
        await self._session.flush()
        return api_key

    async def get(self, api_key_id: str) -> APIKey | None:
        stmt = (
            select(APIKey)
            .options(selectinload(APIKey.user))
            .where(APIKey.id == api_key_id)
        )
        response = await self._session.execute(stmt)
        return response.scalar_one_or_none()

    async def get_by_hash(self, key_hash: str) -> APIKey | None:
        stmt = (
            select(APIKey)
            .options(selectinload(APIKey.user))
            .where(APIKey.key_hash == key_hash)
        )
        response = await self._session.execute(stmt)
        return response.scalar_one_or_none()

    async def list_by_user(self, user_id: str) -> list[APIKey]:
        stmt = (
            select(APIKey)
            .where(APIKey.user_id == user_id)
            .order_by(APIKey.created_at.desc())
        )
        response = await self._session.execute(stmt)
        return list(response.scalars().all())

    async def update_last_used(self, api_key_id: str) -> bool:
        api_key = await self.get(api_key_id)
        if api_key is None:
            return False
        api_key.last_used_at = datetime.now(timezone.utc)
        await self._session.flush()
        return True

    async def deactivate(self, api_key_id: str) -> bool:
        api_key = await self.get(api_key_id)
        if api_key is None:
            return False
        api_key.is_active = False
        await self._session.flush()
        return True

    async def delete(self, api_key_id: str) -> bool:
        api_key = await self.get(api_key_id)
        if api_key is None:
            return False
        await self._session.delete(api_key)
        await self._session.flush()
        return True
