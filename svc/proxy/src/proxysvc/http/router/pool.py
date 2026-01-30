import logging
from typing import List
from typing import Optional

from fastapi import APIRouter
from fastapi import status
from fastapi import Depends
from pydantic import BaseModel

from proxysvc.http.auth.dependencies import get_current_user_id
from proxysvc.http.auth.dependencies import get_current_tenant_id
from proxysvc.http.auth.dependencies import require_scopes
from proxysvc.http.exception import PoolNotFoundException
from proxysvc.http.exception import PoolCreationException
from proxysvc.service import ProxyService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pools", tags=["pools"])

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


class ArmRequest(BaseModel):
    name: str
    metadata: dict = {}


class PoolCreateRequest(BaseModel):
    name: str
    arms: List[ArmRequest]


class ArmResponse(BaseModel):
    id: str
    name: str
    index: int
    is_active: bool = True


class PoolResponse(BaseModel):
    id: str
    name: str
    arms: List[ArmResponse]


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=PoolResponse,
)
async def create_pool(
    body: PoolCreateRequest,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["pool:write"])),
):
    """create a new pool with arms."""
    try:
        service = get_proxy_service()
        arms_data = [{"name": arm.name, "metadata": arm.metadata} for arm in body.arms]
        pool = await service.create_pool(tenant_id, body.name, arms_data)

        logger.info(f"pool created: {pool['id']} by user {user_id}")
        return PoolResponse(
            id=pool["id"],
            name=pool["name"],
            arms=[
                ArmResponse(
                    id=arm["id"],
                    name=arm["name"],
                    index=arm["index"],
                    is_active=arm.get("is_active", True),
                )
                for arm in pool["arms"]
            ],
        )
    except PoolCreationException:
        raise
    except Exception as e:
        logger.error(f"pool creation error: {str(e)}")
        raise PoolCreationException()


@router.get(
    "/{pool_id}",
    status_code=status.HTTP_200_OK,
    response_model=PoolResponse,
)
async def get_pool(
    pool_id: str,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["pool:read"])),
):
    """get pool by id."""
    service = get_proxy_service()
    pool = await service.get_pool(tenant_id, pool_id)

    if pool is None:
        raise PoolNotFoundException(f"pool not found: {pool_id}")

    return PoolResponse(
        id=pool["id"],
        name=pool["name"],
        arms=[
            ArmResponse(
                id=arm["id"],
                name=arm["name"],
                index=arm["index"],
                is_active=arm.get("is_active", True),
            )
            for arm in pool["arms"]
        ],
    )


@router.delete(
    "/{pool_id}",
    status_code=status.HTTP_200_OK,
)
async def delete_pool(
    pool_id: str,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["pool:delete"])),
):
    """delete pool by id."""
    service = get_proxy_service()
    deleted = await service.delete_pool(tenant_id, pool_id)

    if not deleted:
        raise PoolNotFoundException(f"pool not found: {pool_id}")

    logger.info(f"pool deleted: {pool_id} by user {user_id}")
    return {"message": "pool deleted successfully"}


class PoolListResponse(BaseModel):
    pools: List[PoolResponse]
    limit: int
    offset: int


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    response_model=PoolListResponse,
)
async def list_pools(
    limit: int = 100,
    offset: int = 0,
    tenant_id: str = Depends(get_current_tenant_id),
    user_id: str = Depends(get_current_user_id),
    _user=Depends(require_scopes(["pool:read"])),
):
    """list all pools with pagination."""
    service = get_proxy_service()
    pools = await service.list_pools(tenant_id, limit=limit, offset=offset)

    return PoolListResponse(
        pools=[
            PoolResponse(
                id=pool["id"],
                name=pool["name"],
                arms=[
                    ArmResponse(
                        id=arm["id"],
                        name=arm["name"],
                        index=arm["index"],
                        is_active=arm.get("is_active", True),
                    )
                    for arm in pool["arms"]
                ],
            )
            for pool in pools
        ],
        limit=limit,
        offset=offset,
    )
