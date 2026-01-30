import logging
import time
from typing import Optional, Tuple
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from proxysvc.http.auth.operator import auth_operator, token_operator
from proxysvc.http.auth.model import APIKey, User
from proxysvc.http.auth.config import ENDPOINT_SCOPES
from proxysvc.http.exception import BaseAPIException
from proxysvc.http.exception import UnauthorizedException
from proxysvc.http.exception import ForbiddenException
from proxysvc.http.exception import RateLimitedException
from proxysvc.http.exception import InsufficientScopesException
from proxysvc.config import settings

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):

    def __init__(self, app):
        super().__init__(app)
        self.public_paths = {
            "/health",
            "/info",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/auth/register",
            "/auth/login",
            "/auth/refresh",
        }

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        if (
            request.method == "OPTIONS"
        ):  # allow OPTIONS requests (cors preflight) to pass through
            return await call_next(request)

        if self._is_public_path(request.url.path):
            return await call_next(request)

        # bypass authentication in development mode
        # set dummy values for auth
        if settings.runenv == "dev":
            logger.info(
                f"development mode: bypassing auth for {request.method} {request.url.path}"
            )

            request.state.api_key = None
            request.state.user_id = "dev-user"
            request.state.tenant_id = "dev-tenant"
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.debug(
                f"api request (dev mode): {request.method} {request.url.path} | "
                f"time: {process_time:.3f}s | "
                f"status: {response.status_code}"
            )
            return response

        try:
            api_key, user = await self._ensure_request_auth(request)
            if not await self._ensure_scoped_permission(
                api_key, user, request.method, request.url.path
            ):
                raise InsufficientScopesException()

            request.state.api_key = api_key
            request.state.user_id = api_key.user_id if api_key else user.id
            request.state.user = user
            request.state.tenant_id = user.tenant_id

            response = await call_next(request)

            process_time = time.time() - start_time
            auth_method = "API-Key" if api_key else "JWT"
            logger.debug(
                f"api request: {request.method} {request.url.path} | "
                f"auth: {auth_method} | "
                f"user: {request.state.user_id} | "
                f"time: {process_time:.3f}s | "
                f"status: {response.status_code}"
            )

            return response

        except BaseAPIException as e:
            logger.warning(
                f"auth failed: {request.method} {request.url.path} | "
                f"ip: {request.client.host if request.client else 'unknown'} | "
                f"error: {e.detail}"
            )
            return JSONResponse(status_code=e.status_code, content=e.to_dict())
        except Exception as e:
            logger.error(f"auth middleware error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "internal server error"},
            )

    def _is_public_path(self, path: str) -> bool:
        if path in self.public_paths:
            return True

        if path.startswith("/docs") or path.startswith("/redoc"):
            return True

        return False

    # attention: rate limit checks run on every request now; might be a bottleneck.
    #  consider optimizing in case of added latency
    @staticmethod
    async def _ensure_request_auth(request: Request) -> Tuple[Optional[APIKey], User]:
        api_key_header = request.headers.get("X-API-Key")

        if api_key_header:
            api_key = await auth_operator.validate_api_key(api_key_header)
            if api_key:
                if not await auth_operator.check_rate_limit(api_key):
                    raise RateLimitedException(
                        f"rate limit exceeded: {api_key.rate_limit_per_minute} requests per minute"
                    )

                user = await auth_operator.get_user(api_key.user_id)
                if not user or not user.is_active:
                    raise UnauthorizedException("user account is inactive")

                return api_key, user

        auth_header = request.headers.get("Authorization")

        if not auth_header:
            raise UnauthorizedException(
                "authentication required - provide x-api-key header or bearer token"
            )

        if not auth_header.startswith("Bearer "):
            raise UnauthorizedException(
                "invalid authorization header format - use 'bearer <token>' or 'x-api-key: <key>'"
            )

        token = auth_header[7:]  # remove "Bearer " prefix

        if token.startswith("optiq_"):
            api_key = await auth_operator.validate_api_key(token)
            if api_key:
                if not await auth_operator.check_rate_limit(api_key):
                    raise RateLimitedException(
                        f"rate limit exceeded: {api_key.rate_limit_per_minute} requests per minute"
                    )

                user = await auth_operator.get_user(api_key.user_id)
                if not user or not user.is_active:
                    raise UnauthorizedException("user account is inactive")

                return api_key, user

        user_id = token_operator.get_user_id_from_token(token)
        if not user_id:
            raise UnauthorizedException("invalid or expired authentication token")

        user = await auth_operator.get_user(user_id)
        if not user or not user.is_active:
            raise UnauthorizedException("user account is inactive or not found")

        if not await auth_operator.check_user_rate_limit(user):
            raise RateLimitedException(
                "rate limit exceeded - consider using an api key for higher limits"
            )

        return None, user

    @staticmethod
    def _get_required_scope_for_path(method: str, path: str) -> Optional[str]:
        key = (method, path)
        if key in ENDPOINT_SCOPES:
            return ENDPOINT_SCOPES[key]

        for (pattern_method, pattern_path), scope in ENDPOINT_SCOPES.items():
            if method == pattern_method and pattern_path.endswith("*"):
                prefix = pattern_path[:-1]
                if path.startswith(prefix):
                    return scope

        return None

    async def _ensure_scoped_permission(
        self, api_key: Optional[APIKey], user: User, method: str, path: str
    ) -> bool:
        required_scope = self._get_required_scope_for_path(method, path)
        if not required_scope:
            return True
        if api_key:
            return await auth_operator.api_key_has_scope(api_key, required_scope)
        return await auth_operator.user_has_permission(user.id, required_scope)
