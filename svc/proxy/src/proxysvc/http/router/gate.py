import logging
from typing import Optional
from typing import List
from typing import Any

from fastapi import APIRouter
from fastapi import status
from fastapi import Depends
from pydantic import BaseModel

from proxysvc.http.auth.dependencies import get_current_user_id
from proxysvc.http.auth.dependencies import get_current_tenant_id
from proxysvc.http.auth.dependencies import require_scopes
from proxysvc.http.exception import GateNotFoundException
from proxysvc.http.exception import ExperimentNotFoundException
from proxysvc.http.exception import InternalServerException
from proxysvc.service import ProxyService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gates", tags=["gates"])

# module-level proxy service instance, set via set_proxy_service()
_proxy_service: Optional[ProxyService] = None


def set_proxy_service(service: ProxyService) -> None:
    """set the proxy service instance for this router."""
    global _proxy_service
    _proxy_service = service


def get_proxy_service() -> ProxyService:
    """get the proxy service instance."""
    if _proxy_service is None:
        raise RuntimeError("proxy service not initialized")
    return _proxy_service


class RuleRequest(BaseModel):
    key: str
    operator: str
    value: Any
    arm_id: Optional[str] = None


class GateConfigRequest(BaseModel):
    enabled: bool = True
    rollout_percentage: float = 100.0
    default_arm_id: Optional[str] = None
    schedule_start: Optional[str] = None
    schedule_end: Optional[str] = None
    active_hours_start: Optional[str] = None
    active_hours_end: Optional[str] = None
    timezone: str = "UTC"
    rules: List[RuleRequest] = []


class GateConfigResponse(BaseModel):
    experiment: dict
    rules: List[dict] = []
    updated_at: Optional[str] = None
    version: int = 1


@router.post(
    "/{experiment_id}",
    status_code=status.HTTP_201_CREATED,
    response_model=GateConfigResponse,
)
async def create_gate_config(
    experiment_id: str,
    body: GateConfigRequest,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["gate:write"])),
):
    """create feature gate config for an experiment."""
    try:
        service = get_proxy_service()

        config = {
            "enabled": body.enabled,
            "rollout_percentage": body.rollout_percentage,
            "default_arm_id": body.default_arm_id,
            "timezone": body.timezone,
            "rules": [
                {
                    "key": rule.key,
                    "operator": rule.operator,
                    "value": rule.value,
                    "arm": {"committed": {"id": rule.arm_id}} if rule.arm_id else {},
                }
                for rule in body.rules
            ],
        }

        result = await service.create_gate_config(tenant_id, experiment_id, config)

        if result is None:
            raise ExperimentNotFoundException(f"experiment not found: {experiment_id}")

        logger.info(
            f"gate config created for experiment: {experiment_id} by user {user_id}"
        )
        return GateConfigResponse(
            experiment=result.get("experiment", {}),
            rules=result.get("rules", []),
            updated_at=(
                str(result.get("updated_at")) if result.get("updated_at") else None
            ),
            version=result.get("version", 1),
        )
    except ExperimentNotFoundException:
        raise
    except Exception as e:
        logger.error(f"gate config creation error: {str(e)}")
        raise InternalServerException("gate config creation failed")


@router.get(
    "/{experiment_id}",
    status_code=status.HTTP_200_OK,
    response_model=GateConfigResponse,
)
async def get_gate_config(
    experiment_id: str,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["gate:read"])),
):
    """get feature gate config for an experiment."""
    service = get_proxy_service()
    result = await service.get_gate_config(tenant_id, experiment_id)

    if result is None:
        raise GateNotFoundException(
            f"gate config not found for experiment: {experiment_id}"
        )

    return GateConfigResponse(
        experiment=result.get("experiment", {}),
        rules=result.get("rules", []),
        updated_at=str(result.get("updated_at")) if result.get("updated_at") else None,
        version=result.get("version", 1),
    )


@router.put(
    "/{experiment_id}",
    status_code=status.HTTP_200_OK,
    response_model=GateConfigResponse,
)
async def update_gate_config(
    experiment_id: str,
    body: GateConfigRequest,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["gate:write"])),
):
    """update feature gate config for an experiment."""
    try:
        service = get_proxy_service()

        config = {
            "enabled": body.enabled,
            "rollout_percentage": body.rollout_percentage,
            "default_arm_id": body.default_arm_id,
            "timezone": body.timezone,
            "rules": [
                {
                    "key": rule.key,
                    "operator": rule.operator,
                    "value": rule.value,
                    "arm": {"committed": {"id": rule.arm_id}} if rule.arm_id else {},
                }
                for rule in body.rules
            ],
        }

        result = await service.update_gate_config(tenant_id, experiment_id, config)

        if result is None:
            raise GateNotFoundException(
                f"gate config not found for experiment: {experiment_id}"
            )

        logger.info(
            f"gate config updated for experiment: {experiment_id} by user {user_id}"
        )
        return GateConfigResponse(
            experiment=result.get("experiment", {}),
            rules=result.get("rules", []),
            updated_at=(
                str(result.get("updated_at")) if result.get("updated_at") else None
            ),
            version=result.get("version", 1),
        )
    except GateNotFoundException:
        raise
    except Exception as e:
        logger.error(f"gate config update error: {str(e)}")
        raise InternalServerException("gate config update failed")


@router.delete(
    "/{experiment_id}",
    status_code=status.HTTP_200_OK,
)
async def delete_gate_config(
    experiment_id: str,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["gate:write"])),
):
    """delete feature gate config for an experiment."""
    service = get_proxy_service()
    deleted = await service.delete_gate_config(tenant_id, experiment_id)

    if not deleted:
        raise GateNotFoundException(
            f"gate config not found for experiment: {experiment_id}"
        )

    logger.info(
        f"gate config deleted for experiment: {experiment_id} by user {user_id}"
    )
    return {"message": "gate config deleted successfully"}
