import logging
from typing import Optional

from fastapi import APIRouter
from fastapi import status
from fastapi import Depends
from pydantic import BaseModel

from proxysvc.http.auth.dependencies import get_current_user_id
from proxysvc.http.auth.dependencies import get_current_tenant_id
from proxysvc.http.auth.dependencies import require_scopes
from proxysvc.http.exception import ExperimentNotFoundException
from proxysvc.http.exception import ExperimentCreationException
from proxysvc.http.exception import InternalServerException
from proxysvc.service import ProxyService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])

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


class FeatureGateRequest(BaseModel):
    enabled: bool = True
    rollout_percentage: float = 100.0
    default_arm_id: Optional[str] = None
    timezone: str = "UTC"


class ExperimentCreateRequest(BaseModel):
    name: str
    pool_id: str
    policy: str
    policy_params: dict = {}
    enabled: bool = True
    feature_gate: Optional[FeatureGateRequest] = None


class ExperimentUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    policy_params: Optional[dict] = None


class ExperimentResponse(BaseModel):
    id: str
    name: str
    pool_id: str
    policy: str
    policy_params: dict
    enabled: bool


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=ExperimentResponse,
)
async def create_experiment(
    body: ExperimentCreateRequest,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["experiment:write"])),
):
    """create a new experiment."""
    try:
        service = get_proxy_service()

        feature_gate_config = None
        if body.feature_gate:
            feature_gate_config = {
                "enabled": body.feature_gate.enabled,
                "rollout_percentage": body.feature_gate.rollout_percentage,
                "default_arm_id": body.feature_gate.default_arm_id,
                "timezone": body.feature_gate.timezone,
            }

        experiment = await service.create_experiment(
            tenant_id=tenant_id,
            name=body.name,
            pool_id=body.pool_id,
            policy=body.policy,
            policy_params=body.policy_params,
            enabled=body.enabled,
            feature_gate_config=feature_gate_config,
        )

        logger.info(f"experiment created: {experiment['id']} by user {user_id}")
        return ExperimentResponse(
            id=experiment["id"],
            name=experiment["name"],
            pool_id=experiment["pool_id"],
            policy=experiment["policy"],
            policy_params=experiment.get("policy_params", {}),
            enabled=experiment.get("enabled", True),
        )
    except ExperimentCreationException:
        raise
    except Exception as e:
        logger.error(f"experiment creation error: {str(e)}")
        raise ExperimentCreationException()


@router.get(
    "/{experiment_id}",
    status_code=status.HTTP_200_OK,
    response_model=ExperimentResponse,
)
async def get_experiment(
    experiment_id: str,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["experiment:read"])),
):
    """get experiment by id."""
    service = get_proxy_service()
    experiment = await service.get_experiment(tenant_id, experiment_id)

    if experiment is None:
        raise ExperimentNotFoundException(f"experiment not found: {experiment_id}")

    return ExperimentResponse(
        id=experiment["id"],
        name=experiment["name"],
        pool_id=experiment["pool_id"],
        policy=experiment["policy"],
        policy_params=experiment.get("policy_params", {}),
        enabled=experiment.get("enabled", True),
    )


@router.patch(
    "/{experiment_id}",
    status_code=status.HTTP_200_OK,
    response_model=ExperimentResponse,
)
async def update_experiment(
    experiment_id: str,
    body: ExperimentUpdateRequest,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["experiment:write"])),
):
    """update experiment by id."""
    try:
        service = get_proxy_service()

        kwargs = {}
        if body.enabled is not None:
            kwargs["enabled"] = body.enabled
        if body.policy_params is not None:
            kwargs["policy_params"] = body.policy_params

        experiment = await service.update_experiment(tenant_id, experiment_id, **kwargs)

        if experiment is None:
            raise ExperimentNotFoundException(f"experiment not found: {experiment_id}")

        logger.info(f"experiment updated: {experiment_id} by user {user_id}")
        return ExperimentResponse(
            id=experiment["id"],
            name=experiment["name"],
            pool_id=experiment["pool_id"],
            policy=experiment["policy"],
            policy_params=experiment.get("policy_params", {}),
            enabled=experiment.get("enabled", True),
        )
    except ExperimentNotFoundException:
        raise
    except Exception as e:
        logger.error(f"experiment update error: {str(e)}")
        raise InternalServerException("experiment update failed")


@router.delete(
    "/{experiment_id}",
    status_code=status.HTTP_200_OK,
)
async def delete_experiment(
    experiment_id: str,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["experiment:delete"])),
):
    """delete experiment by id."""
    service = get_proxy_service()
    deleted = await service.delete_experiment(tenant_id, experiment_id)

    if not deleted:
        raise ExperimentNotFoundException(f"experiment not found: {experiment_id}")

    logger.info(f"experiment deleted: {experiment_id} by user {user_id}")
    return {"message": "experiment deleted successfully"}


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentResponse]
    limit: int
    offset: int


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    response_model=ExperimentListResponse,
)
async def list_experiments(
    limit: int = 100,
    offset: int = 0,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["experiment:read"])),
):
    """list all experiments with pagination."""
    service = get_proxy_service()
    experiments = await service.list_experiments(tenant_id, limit=limit, offset=offset)

    return ExperimentListResponse(
        experiments=[
            ExperimentResponse(
                id=exp["id"],
                name=exp["name"],
                pool_id=exp["pool_id"],
                policy=exp["policy"],
                policy_params=exp.get("policy_params", {}),
                enabled=exp.get("enabled", True),
            )
            for exp in experiments
        ],
        limit=limit,
        offset=offset,
    )
