from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=ZoneInfo("UTC")),
        description="Timestamp of the last update",
    )
    version: int = Field(default=0, description="Version of the configuration")


class BaseArmModel(BaseModel):
    name: Optional[str] = Field(default=None, description="Name of the arm")
    id: Optional[str] = Field(default=None, description="Unique identifier for the arm")
    index: Optional[int] = Field(
        default=None, description="Index of the arm in the pool"
    )


class ArmConfig(BaseModel):
    committed: BaseArmModel = BaseArmModel()
