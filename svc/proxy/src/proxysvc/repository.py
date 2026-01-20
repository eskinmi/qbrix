from __future__ import annotations

from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from qbrixstore.postgres.models import Pool, Arm, Experiment, FeatureGate

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


class PoolRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, name: str, arms: list[dict]) -> Pool:
        pool = Pool(name=name)
        for i, arm_data in enumerate(arms):
            arm = Arm(
                name=arm_data["name"], index=i, metadata_=arm_data.get("metadata", {})
            )
            pool.arms.append(arm)
        self._session.add(pool)
        await self._session.flush()
        return pool

    async def get(self, pool_id: str) -> Pool | None:
        stmt = select(Pool).options(selectinload(Pool.arms)).where(Pool.id == pool_id)
        response = await self._session.execute(stmt)
        return response.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Pool | None:
        stmt = select(Pool).options(selectinload(Pool.arms)).where(Pool.name == name)
        response = await self._session.execute(stmt)
        return response.scalar_one_or_none()

    async def delete(self, pool_id: str) -> bool:
        pool = await self.get(pool_id)
        if pool is None:
            return False
        await self._session.delete(pool)
        await self._session.flush()
        return True


class ExperimentRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(
        self,
        name: str,
        pool_id: str,
        protocol: str,
        protocol_params: dict,
        enabled: bool,
        feature_gate_config: dict | None = None,
    ) -> Experiment:
        experiment = Experiment(
            name=name,
            pool_id=pool_id,
            protocol=protocol,
            protocol_params=protocol_params,
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
            .where(Experiment.id == experiment_id)
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
            .where(Experiment.name == name)
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
