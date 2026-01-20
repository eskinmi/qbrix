from datetime import datetime, time, timezone
from uuid import uuid4
from sqlalchemy import String, Boolean, Float, Integer, ForeignKey, JSON, DateTime, Time
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Pool(Base):
    __tablename__ = "pools"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=lambda: uuid4().hex
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    arms: Mapped[list["Arm"]] = relationship(
        "Arm", back_populates="pool", cascade="all, delete-orphan"
    )
    experiments: Mapped[list["Experiment"]] = relationship(
        "Experiment", back_populates="pool"
    )


class Arm(Base):
    __tablename__ = "arms"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=lambda: uuid4().hex
    )
    pool_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("pools.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    pool: Mapped["Pool"] = relationship("Pool", back_populates="arms")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=lambda: uuid4().hex
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    pool_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("pools.id"), nullable=False
    )
    protocol: Mapped[str] = mapped_column(String(64), nullable=False)
    protocol_params: Mapped[dict] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    pool: Mapped["Pool"] = relationship("Pool", back_populates="experiments")
    feature_gate: Mapped["FeatureGate"] = relationship(
        "FeatureGate",
        back_populates="experiment",
        uselist=False,
        cascade="all, delete-orphan",
    )


class FeatureGate(Base):
    __tablename__ = "feature_gates"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=lambda: uuid4().hex
    )
    experiment_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("experiments.id"), nullable=False, unique=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    rollout_percentage: Mapped[float] = mapped_column(Float, default=100.0)
    default_arm_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("arms.id"), nullable=True
    )
    schedule_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    schedule_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    active_hours_start: Mapped[time | None] = mapped_column(Time, nullable=True)
    active_hours_end: Mapped[time | None] = mapped_column(Time, nullable=True)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    rules: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="feature_gate"
    )
    default_arm: Mapped["Arm | None"] = relationship(
        "Arm", foreign_keys=[default_arm_id]
    )
