from __future__ import annotations

import logging
from datetime import datetime
from datetime import timedelta
from typing import Any

from jose import JWTError
from jose import jwt

from proxysvc.http.auth.service import AuthService
from proxysvc.config import settings

logger = logging.getLogger(__name__)

# module-level instances, initialized via init_operators()
auth_operator: AuthOperator | None = None
token_operator: TokenOperator | None = None


class AuthOperator:
    """thin wrapper around AuthService for HTTP layer compatibility."""

    def __init__(self, auth_service: AuthService):
        self._service = auth_service

    async def create_user(
        self,
        email: str,
        password: str,
        tenant_id: str | None = None,
        plan_tier: str = "free",
        role: str = "member",
    ):
        """create a new user."""
        user_dict = await self._service.register_user(
            email=email,
            password=password,
            tenant_id=tenant_id,
            plan_tier=plan_tier,
            role=role,
        )
        return _UserWrapper(user_dict)

    async def get_user(self, user_id: str):
        """get user by id."""
        user_dict = await self._service.get_user(user_id)
        if user_dict is None:
            return None
        return _UserWrapper(user_dict)

    async def get_user_by_email(self, email: str):
        """get user by email."""
        user_dict = await self._service.get_user_by_email(email)
        if user_dict is None:
            return None
        return _UserWrapper(user_dict)

    async def authenticate_user(self, email: str, password: str):
        """authenticate user with email and password."""
        user_dict = await self._service.authenticate_user(email, password)
        if user_dict is None:
            return None
        return _UserWrapper(user_dict)

    async def create_api_key(
        self,
        user_id: str,
        name: str = "Default API Key",
        scopes: list[str] | None = None,
    ) -> tuple:
        """create api key for user. returns (api_key_wrapper, plain_key)."""
        api_key_dict, plain_key = await self._service.create_api_key(
            user_id=user_id,
            name=name,
            scopes=scopes,
        )
        return _APIKeyWrapper(api_key_dict), plain_key

    async def get_api_key(self, api_key_id: str):
        """get api key by id."""
        api_key_dict = await self._service.get_api_key(api_key_id)
        if api_key_dict is None:
            return None
        return _APIKeyWrapper(api_key_dict)

    async def validate_api_key(self, plain_key: str):
        """validate api key and return wrapper if valid."""
        api_key_dict = await self._service.validate_api_key(plain_key)
        if api_key_dict is None:
            return None
        return _APIKeyWrapper(api_key_dict)

    async def get_user_api_keys(self, user_id: str) -> list:
        """get all api keys for user."""
        api_keys = await self._service.list_user_api_keys(user_id)
        return [_APIKeyWrapper(k) for k in api_keys]

    async def deactivate_api_key(self, api_key_id: str, user_id: str) -> bool:
        """deactivate api key."""
        return await self._service.deactivate_api_key(api_key_id, user_id)

    async def check_rate_limit(self, api_key) -> bool:
        """check rate limit for api key."""
        allowed, _, _ = await self._service.check_rate_limit(api_key.id)
        return allowed

    async def check_user_rate_limit(self, user) -> bool:
        """check rate limit for user."""
        allowed, _, _ = await self._service.check_user_rate_limit(user.id)
        return allowed

    async def get_api_key_usage(self, api_key) -> dict:
        """get api key usage stats."""
        return await self._service.get_api_key_usage(api_key.id)

    async def assign_role_to_user(self, user_id: str, role: str) -> bool:
        """assign role to user."""
        result = await self._service.assign_role(user_id, role)
        return result is not None

    async def list_users(self, limit: int = 100, offset: int = 0) -> list:
        """list all users with pagination."""
        users = await self._service.list_users(limit=limit, offset=offset)
        return [_UserWrapper(u) for u in users]

    async def user_has_permission(self, user_id: str, required_scope: str) -> bool:
        """check if user has required scope."""
        return await self._service.check_user_permission(user_id, required_scope)

    @staticmethod
    async def api_key_has_scope(api_key, required_scope: str) -> bool:
        """check if api key has required scope."""
        return required_scope in api_key.scopes or "system:admin" in api_key.scopes

    @staticmethod
    def get_scopes_for_role(role: str) -> list[str]:
        """get scopes for a given role."""
        return AuthService.get_scopes_for_role(role)


class TokenOperator:
    """stateless JWT token handling."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        self._secret_key = secret_key
        self._algorithm = algorithm
        self._access_token_expire_minutes = access_token_expire_minutes
        self._refresh_token_expire_days = refresh_token_expire_days

    def create_access_token(self, user) -> str:
        """create access token for user."""
        expire = datetime.utcnow() + timedelta(
            minutes=self._access_token_expire_minutes
        )
        payload = {
            "sub": user.id,
            "tenant_id": user.tenant_id,
            "email": user.email,
            "role": user.role,
            "plan_tier": user.plan_tier,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }
        token = jwt.encode(payload, self._secret_key, algorithm=self._algorithm)
        logger.info(f"created access token for user {user.id}")
        return token

    def create_refresh_token(self, user) -> str:
        """create refresh token for user."""
        expire = datetime.utcnow() + timedelta(days=self._refresh_token_expire_days)
        payload = {
            "sub": user.id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
        }
        token = jwt.encode(payload, self._secret_key, algorithm=self._algorithm)
        logger.info(f"created refresh token for user {user.id}")
        return token

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """verify and decode token."""
        try:
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
                options={"verify_exp": False},
            )
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                logger.warning("token has expired")
                return None
            return payload
        except JWTError as e:
            logger.warning(f"jwt validation error: {e}")
            return None
        except Exception as e:
            logger.error(f"unexpected error during token validation: {e}")
            return None

    def get_user_id_from_token(self, token: str) -> str | None:
        """extract user id from access token."""
        payload = self.verify_token(token)
        if payload and payload.get("type") == "access":
            return payload.get("sub")
        return None

    async def refresh_access_token(self, refresh_token: str) -> str | None:
        """refresh access token using refresh token."""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None

        user_id = payload.get("sub")
        if not user_id:
            return None

        user = await auth_operator.get_user(user_id)
        if not user or not user.is_active:
            logger.warning(
                f"user {user_id} is inactive or not found, cannot refresh token"
            )
            return None

        new_access_token = self.create_access_token(user)
        logger.info(f"access token refreshed for user {user_id}")
        return new_access_token


class _UserWrapper:
    """wrapper to provide attribute access to user dict for backward compatibility."""

    def __init__(self, data: dict):
        self._data = data

    @property
    def id(self) -> str:
        return self._data["id"]

    @property
    def tenant_id(self) -> str:
        return self._data["tenant_id"]

    @property
    def email(self) -> str:
        return self._data["email"]

    @property
    def plan_tier(self) -> str:
        return self._data["plan_tier"]

    @property
    def role(self) -> str:
        return self._data["role"]

    @property
    def is_active(self) -> bool:
        return self._data.get("is_active", True)

    @property
    def created_at(self) -> float:
        return self._data.get("created_at", 0.0)

    @property
    def updated_at(self) -> float:
        return self._data.get("updated_at", 0.0)


class _APIKeyWrapper:
    """wrapper to provide attribute access to api key dict for backward compatibility."""

    def __init__(self, data: dict):
        self._data = data

    @property
    def id(self) -> str:
        return self._data["id"]

    @property
    def user_id(self) -> str:
        return self._data["user_id"]

    @property
    def name(self) -> str:
        return self._data["name"]

    @property
    def rate_limit_per_minute(self) -> int:
        return self._data["rate_limit_per_minute"]

    @property
    def scopes(self) -> list[str]:
        return self._data.get("scopes", [])

    @property
    def is_active(self) -> bool:
        return self._data.get("is_active", True)

    @property
    def created_at(self) -> float:
        return self._data.get("created_at", 0.0)

    @property
    def last_used_at(self) -> float | None:
        return self._data.get("last_used_at")


def init_operators(auth_service: AuthService) -> None:
    """initialize module-level operators with auth service instance."""
    global auth_operator, token_operator

    auth_operator = AuthOperator(auth_service)
    token_operator = TokenOperator(
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        access_token_expire_minutes=settings.jwt_access_token_expire_minutes,
        refresh_token_expire_days=settings.jwt_refresh_token_expire_days,
    )
    logger.info("initialized auth and token operators")
