from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi import status
from fastapi import Depends
from fastapi import Query
from pydantic import BaseModel

from qbrixstore.clickhouse.client import ClickHouseClient
from qbrixstore.config import ClickHouseSettings

from proxysvc.http.auth.dependencies import get_current_tenant_id
from proxysvc.http.auth.dependencies import get_current_user_id
from proxysvc.http.auth.dependencies import require_scopes
from proxysvc.http.exception import InternalServerException
from proxysvc.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ee/analytics", tags=["ee-analytics"])

_clickhouse_client: Optional[ClickHouseClient] = None


def get_clickhouse_client() -> ClickHouseClient:
    """get or create clickhouse client."""
    global _clickhouse_client
    if _clickhouse_client is None:
        ch_settings = ClickHouseSettings(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            user=settings.clickhouse_user,
            password=settings.clickhouse_password,
            database=settings.clickhouse_database,
        )
        _clickhouse_client = ClickHouseClient(ch_settings)
        _clickhouse_client.connect()
    return _clickhouse_client


class ExperimentStatsResponse(BaseModel):
    experiment_id: str
    total_selections: int
    default_selections: int
    unique_contexts: int
    first_selection_ms: int | None
    last_selection_ms: int | None
    total_feedback: int
    avg_reward: float | None
    min_reward: float | None
    max_reward: float | None


class TimeseriesPointResponse(BaseModel):
    timestamp_ms: int
    selections: int
    default_selections: int


class TimeseriesResponse(BaseModel):
    experiment_id: str
    interval_ms: int
    data: list[TimeseriesPointResponse]


class ArmStatsResponse(BaseModel):
    arm_index: int
    arm_name: str
    selections: int
    feedback_count: int
    avg_reward: float | None


class ArmAnalyticsResponse(BaseModel):
    experiment_id: str
    arms: list[ArmStatsResponse]


@router.get(
    "/experiments/{experiment_id}",
    status_code=status.HTTP_200_OK,
    response_model=ExperimentStatsResponse,
)
async def get_experiment_stats(
    experiment_id: str,
    start_ms: Optional[int] = Query(None, description="start timestamp in ms"),
    end_ms: Optional[int] = Query(None, description="end timestamp in ms"),
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["analytics:read"])),
):
    """get aggregated stats for an experiment."""
    try:
        client = get_clickhouse_client()
        stats = client.get_experiment_stats(
            tenant_id=tenant_id,
            experiment_id=experiment_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )

        return ExperimentStatsResponse(
            experiment_id=experiment_id,
            total_selections=stats["total_selections"],
            default_selections=stats["default_selections"],
            unique_contexts=stats["unique_contexts"],
            first_selection_ms=stats["first_selection_ms"],
            last_selection_ms=stats["last_selection_ms"],
            total_feedback=stats["total_feedback"],
            avg_reward=stats["avg_reward"],
            min_reward=stats["min_reward"],
            max_reward=stats["max_reward"],
        )
    except Exception as e:
        logger.error(f"failed to get experiment stats: {e}")
        raise InternalServerException("failed to get experiment stats")


@router.get(
    "/experiments/{experiment_id}/timeseries",
    status_code=status.HTTP_200_OK,
    response_model=TimeseriesResponse,
)
async def get_experiment_timeseries(
    experiment_id: str,
    interval_ms: int = Query(3600000, description="bucket interval in ms (default 1 hour)"),
    start_ms: Optional[int] = Query(None, description="start timestamp in ms"),
    end_ms: Optional[int] = Query(None, description="end timestamp in ms"),
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["analytics:read"])),
):
    """get time series data for an experiment."""
    try:
        client = get_clickhouse_client()
        data = client.get_experiment_timeseries(
            tenant_id=tenant_id,
            experiment_id=experiment_id,
            interval_ms=interval_ms,
            start_ms=start_ms,
            end_ms=end_ms,
        )

        return TimeseriesResponse(
            experiment_id=experiment_id,
            interval_ms=interval_ms,
            data=[
                TimeseriesPointResponse(
                    timestamp_ms=point["timestamp_ms"],
                    selections=point["selections"],
                    default_selections=point["default_selections"],
                )
                for point in data
            ],
        )
    except Exception as e:
        logger.error(f"failed to get experiment timeseries: {e}")
        raise InternalServerException("failed to get experiment timeseries")


@router.get(
    "/experiments/{experiment_id}/arms",
    status_code=status.HTTP_200_OK,
    response_model=ArmAnalyticsResponse,
)
async def get_arm_analytics(
    experiment_id: str,
    start_ms: Optional[int] = Query(None, description="start timestamp in ms"),
    end_ms: Optional[int] = Query(None, description="end timestamp in ms"),
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["analytics:read"])),
):
    """get per-arm statistics for an experiment."""
    try:
        client = get_clickhouse_client()
        arms = client.get_arm_stats(
            tenant_id=tenant_id,
            experiment_id=experiment_id,
            start_ms=start_ms,
            end_ms=end_ms,
        )

        return ArmAnalyticsResponse(
            experiment_id=experiment_id,
            arms=[
                ArmStatsResponse(
                    arm_index=arm["arm_index"],
                    arm_name=arm["arm_name"],
                    selections=arm["selections"],
                    feedback_count=arm["feedback_count"],
                    avg_reward=arm["avg_reward"],
                )
                for arm in arms
            ],
        )
    except Exception as e:
        logger.error(f"failed to get arm analytics: {e}")
        raise InternalServerException("failed to get arm analytics")
