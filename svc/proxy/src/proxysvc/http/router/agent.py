import logging
from typing import Optional, List, Union

from fastapi import APIRouter
from fastapi import status
from fastapi import Depends
from pydantic import BaseModel

from proxysvc.http.auth.dependencies import get_current_user_id
from proxysvc.http.auth.dependencies import require_scopes
from proxysvc.http.exception import SelectionException
from proxysvc.http.exception import FeedbackException
from proxysvc.service import ProxyService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

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


class ContextModel(BaseModel):
    id: str
    vector: Optional[List[Union[int, float]]] = None
    metadata: Optional[dict] = None


class AgentSelectRequest(BaseModel):
    experiment_id: str
    context: ContextModel


class AgentFeedbackRequest(BaseModel):
    request_id: str
    reward: Union[int, float]



@router.post(
    "/select",
    status_code=status.HTTP_200_OK,
)
async def agent_select(
    body: AgentSelectRequest,
    user_id: str = Depends(get_current_user_id),  # noqa
    _user=Depends(require_scopes(["agent:read"])),
):
    try:
        svc = get_proxy_service()
        response = await svc.select(
            experiment_id=body.experiment_id,
            context_id=body.context.id,
            context_vector=body.context.vector,
            context_metadata=body.context.metadata
        )

        logger.info(f"agent selected for experiment: {body.experiment_id}")

        return response
    except SelectionException:
        raise
    except Exception as e:
        logger.error(f"agent select: {str(e)}")
        raise SelectionException()


@router.post(
    "/feedback",
    status_code=status.HTTP_201_CREATED
)
async def agent_feedback(
    body: AgentFeedbackRequest,
    user_id: str = Depends(get_current_user_id),  # noqa
    _user=Depends(require_scopes(["agent:write"])),
):
    try:
        svc = get_proxy_service()
        accepted = await svc.feed(
            request_id=body.request_id,
            reward=body.reward,
        )
        logger.info(f"agent feedback received for request: {body.request_id}")
        return {"accepted": accepted}
    except FeedbackException:
        raise
    except Exception as e:
        logger.error(f"agent feed: {str(e)}")
        raise FeedbackException()