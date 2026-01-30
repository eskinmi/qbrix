import logging
import time
from typing import List

from fastapi import Request
from fastapi import Depends

from proxysvc.http.auth.operator import auth_operator
from proxysvc.http.auth.operator import token_operator
from proxysvc.http.exception import UnauthorizedException
from proxysvc.http.exception import ForbiddenException
from proxysvc.http.exception import UserNotFoundException
from proxysvc.http.exception import InvalidTokenException
from proxysvc.http.exception import InsufficientScopesException
from proxysvc.config import settings

logger = logging.getLogger(__name__)


async def get_current_user_id(request: Request) -> str:
    """extract user id from request state or authorization header."""
    if hasattr(request.state, "user_id"):
        return request.state.user_id

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise UnauthorizedException(
            "authentication required - provide api key or jwt token"
        )

    if not auth_header.startswith("Bearer "):
        raise UnauthorizedException("invalid authorization header format")

    token = auth_header[7:]
    user_id = token_operator.get_user_id_from_token(token)

    if not user_id:
        raise InvalidTokenException("invalid or expired jwt token")

    return user_id


async def get_current_tenant_id(request: Request) -> str:
    """extract tenant id from request state."""
    if hasattr(request.state, "tenant_id"):
        return request.state.tenant_id

    raise UnauthorizedException(
        "tenant context not available - ensure authentication is configured"
    )


class _DevUserWrapper:
    """wrapper for dev user to provide attribute access."""

    def __init__(self):
        self.id = "dev-user"
        self.tenant_id = "dev-tenant"
        self.email = "dev@optiq.io"
        self.role = "admin"
        self.plan_tier = "enterprise"
        self.created_at = time.time()
        self.is_active = True


async def get_current_user(user_id: str = Depends(get_current_user_id)):
    """get current user from auth service."""
    if settings.runenv == "dev" and user_id == "dev-user":
        return _DevUserWrapper()

    user = await auth_operator.get_user(user_id)
    if not user:
        raise UserNotFoundException()
    return user


async def get_current_active_user(user=Depends(get_current_user)):
    """ensure current user is active."""
    if not user.is_active:
        raise ForbiddenException("user account is inactive")
    return user


async def require_admin_user(user=Depends(get_current_active_user)):
    """require admin role."""
    if user.role != "admin":
        raise ForbiddenException("admin role required")
    return user


async def require_member_or_above(user=Depends(get_current_active_user)):
    """require member or admin role."""
    allowed_roles = {"admin", "member"}
    if user.role not in allowed_roles:
        raise ForbiddenException("member role or above required")
    return user


def require_scopes(required_scopes: List[str]):
    """dependency factory that requires specific scopes."""

    async def scope_dependency(user=Depends(get_current_active_user)):
        user_scopes = auth_operator.get_scopes_for_role(user.role)

        for scope in required_scopes:
            if scope not in user_scopes:
                raise InsufficientScopesException(
                    f"required scope '{scope}' not available"
                )
        return user

    return scope_dependency


def require_any_scope(required_scopes: List[str]):
    """dependency factory that requires any one of the specified scopes."""

    async def scope_dependency(user=Depends(get_current_active_user)):
        user_scopes = auth_operator.get_scopes_for_role(user.role)

        if not any(scope in user_scopes for scope in required_scopes):
            raise InsufficientScopesException(
                f"at least one of these scopes required: {', '.join(required_scopes)}"
            )
        return user

    return scope_dependency
