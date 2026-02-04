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

router = APIRouter(prefix="/ee/traces", tags=["ee-traces"])

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


class SelectionTraceResponse(BaseModel):
    tenant_id: str
    experiment_id: str
    request_id: str
    arm_id: str
    arm_name: str
    arm_index: int
    is_default: bool
    context_id: str
    timestamp_ms: int
    policy: str


class FeedbackTraceResponse(BaseModel):
    tenant_id: str
    experiment_id: str
    request_id: str
    arm_index: int
    reward: float
    context_id: str
    timestamp_ms: int


class TraceResponse(BaseModel):
    selection: SelectionTraceResponse | None
    feedback: FeedbackTraceResponse | None


class TracesListResponse(BaseModel):
    traces: list[SelectionTraceResponse]
    limit: int
    offset: int


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    response_model=TracesListResponse,
)
async def list_traces(
    experiment_id: Optional[str] = Query(None, description="filter by experiment id"),
    limit: int = Query(100, le=1000, description="max results to return"),
    offset: int = Query(0, ge=0, description="offset for pagination"),
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["trace:read"])),
):
    """list selection traces with optional filtering."""
    try:
        client = get_clickhouse_client()
        results = client.query_selection_events(
            tenant_id=tenant_id,
            experiment_id=experiment_id,
            limit=limit,
            offset=offset,
        )

        traces = [
            SelectionTraceResponse(
                tenant_id=r["tenant_id"],
                experiment_id=r["experiment_id"],
                request_id=r["request_id"],
                arm_id=r["arm_id"],
                arm_name=r["arm_name"],
                arm_index=r["arm_index"],
                is_default=r["is_default"],
                context_id=r["context_id"],
                timestamp_ms=r["timestamp_ms"],
                policy=r["policy"],
            )
            for r in results
        ]

        return TracesListResponse(
            traces=traces,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"failed to list traces: {e}")
        raise InternalServerException("failed to list traces")


@router.get(
    "/{request_id}",
    status_code=status.HTTP_200_OK,
    response_model=TraceResponse,
)
async def get_trace(
    request_id: str,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["trace:read"])),
):
    """get full trace for a request (selection + feedback)."""
    try:
        client = get_clickhouse_client()

        selection_results = client.query_selection_events(
            tenant_id=tenant_id,
            request_id=request_id,
            limit=1,
        )

        feedback_results = client.query_feedback_events(
            tenant_id=tenant_id,
            request_id=request_id,
            limit=1,
        )

        selection = None
        if selection_results:
            r = selection_results[0]
            selection = SelectionTraceResponse(
                tenant_id=r["tenant_id"],
                experiment_id=r["experiment_id"],
                request_id=r["request_id"],
                arm_id=r["arm_id"],
                arm_name=r["arm_name"],
                arm_index=r["arm_index"],
                is_default=r["is_default"],
                context_id=r["context_id"],
                timestamp_ms=r["timestamp_ms"],
                policy=r["policy"],
            )

        feedback = None
        if feedback_results:
            r = feedback_results[0]
            feedback = FeedbackTraceResponse(
                tenant_id=r["tenant_id"],
                experiment_id=r["experiment_id"],
                request_id=r["request_id"],
                arm_index=r["arm_index"],
                reward=r["reward"],
                context_id=r["context_id"],
                timestamp_ms=r["timestamp_ms"],
            )

        return TraceResponse(
            selection=selection,
            feedback=feedback,
        )
    except Exception as e:
        logger.error(f"failed to get trace: {e}")
        raise InternalServerException("failed to get trace")
