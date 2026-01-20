from typing import Optional
from datetime import datetime, time
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from .base import BaseConfig
from .base import ArmConfig


class RolloutConfig(BaseModel):
    percentage: float = Field(
        default=100.0, description="Percentage of users to be included in the rollout"
    )

    def is_in_rollout(self, identifier: str) -> bool:
        return (abs(hash(identifier)) % 100) < self.percentage


class ActiveHoursConfig(BaseModel):
    start: Optional[time] = Field(
        default=None, description="Start time of the active hours"
    )
    end: Optional[time] = Field(
        default=None, description="End time of the active hours"
    )
    timezone: ZoneInfo = Field(
        default_factory=lambda: ZoneInfo("UTC"),
        description="Timezone for the active hours",
    )

    def is_in_active_hours(self):
        now = datetime.now(tz=self.timezone).time()

        if self.start is None or self.end is None:
            return True

        if self.start <= self.end:
            return self.start <= now <= self.end
        else:
            return now >= self.start or now <= self.end


class ActivePeriodConfig(BaseModel):
    start: Optional[datetime] = Field(
        default=None, description="Start datetime of the active period"
    )
    end: Optional[datetime] = Field(
        default=None, description="End datetime of the active period"
    )
    timezone: ZoneInfo = Field(
        default_factory=lambda: ZoneInfo("UTC"),
        description="Timezone for the active period",
    )

    def is_in_active_period(self):
        now = datetime.now(tz=self.timezone)
        if self.start and now < self.start.astimezone(self.timezone):
            return False
        if self.end and now > self.end.astimezone(self.timezone):
            return False
        return True


class ScheduleConfig(BaseModel):
    hour: ActiveHoursConfig = ActiveHoursConfig()
    period: ActivePeriodConfig = ActivePeriodConfig()

    def is_in_active_schedule(self):
        return self.hour.is_in_active_hours() and self.period.is_in_active_period()


class ExperimentConfig(BaseConfig):
    experiment_id: str = Field(..., description="Unique identifier for the experiment")
    enabled: bool = Field(
        default=True, description="Flag to enable or disable the experiment"
    )
    arm: ArmConfig = ArmConfig()
    rollout: RolloutConfig = RolloutConfig()
    schedule: ScheduleConfig = ScheduleConfig()
