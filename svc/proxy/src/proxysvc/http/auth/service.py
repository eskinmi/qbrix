from __future__ import annotations

import hashlib
import logging
import secrets
import time

import bcrypt

from qbrixstore.postgres.session import get_session
from qbrixstore.redis.client import RedisClient

from proxysvc.repository import TenantRepository
from proxysvc.repository import UserRepository
from proxysvc.repository import APIKeyRepository
from proxysvc.http.auth.config import ROLE_SCOPES
from proxysvc.http.auth.config import PLAN_LIMITS

logger = logging.getLogger(__name__)


class AuthService:
    """core authentication service with business logic for user and api key management."""

    def __init__(self, redis: RedisClient):
        self._redis = redis
        self._api_key_prefix = "optiq_"

    # user management

    async def register_user(
        self,
        email: str,
        password: str,
        tenant_id: str | None = None,
        plan_tier: str = "free",
        role: str = "member",
    ) -> dict:
        """register a new user with email and password.

        if tenant_id is not provided, creates a new tenant for the user
        using the email prefix as the tenant slug.
        """
        async with get_session() as session:
            user_repo = UserRepository(session)

            existing = await user_repo.get_by_email(email)
            if existing:
                raise ValueError(f"user with email {email} already exists")

            # create tenant if not provided
            if tenant_id is None:
                tenant_repo = TenantRepository(session)
                slug = email.split("@")[0].lower().replace(".", "-")
                # ensure unique slug
                base_slug = slug
                counter = 1
                while await tenant_repo.get_by_slug(slug):
                    slug = f"{base_slug}-{counter}"
                    counter += 1
                tenant = await tenant_repo.create(
                    name=f"{email}'s Workspace",
                    slug=slug,
                )
                tenant_id = tenant.id
                logger.info(f"created tenant {tenant_id} with slug {slug}")

            password_hash = bcrypt.hashpw(
                password.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")

            user = await user_repo.create(
                email=email,
                password_hash=password_hash,
                tenant_id=tenant_id,
                plan_tier=plan_tier,
                role=role,
            )
            logger.info(f"created user {user.id} with email {email}")
            return self._user_to_dict(user)

    async def authenticate_user(self, email: str, password: str) -> dict | None:
        """authenticate user with email and password."""
        async with get_session() as session:
            repo = UserRepository(session)
            user = await repo.get_by_email(email)

            if not user or not user.is_active:
                return None

            if bcrypt.checkpw(
                password.encode("utf-8"), user.password_hash.encode("utf-8")
            ):
                return self._user_to_dict(user)
            return None

    async def get_user(self, user_id: str) -> dict | None:
        """get user by id."""
        async with get_session() as session:
            repo = UserRepository(session)
            user = await repo.get(user_id)
            if user is None:
                return None
            return self._user_to_dict(user)

    async def get_user_by_email(self, email: str) -> dict | None:
        """get user by email."""
        async with get_session() as session:
            repo = UserRepository(session)
            user = await repo.get_by_email(email)
            if user is None:
                return None
            return self._user_to_dict(user)

    async def update_user(
        self,
        user_id: str,
        plan_tier: str | None = None,
        is_active: bool | None = None,
    ) -> dict | None:
        """update user fields."""
        async with get_session() as session:
            repo = UserRepository(session)
            kwargs = {}
            if plan_tier is not None:
                kwargs["plan_tier"] = plan_tier
            if is_active is not None:
                kwargs["is_active"] = is_active
            user = await repo.update(user_id, **kwargs)
            if user is None:
                return None
            return self._user_to_dict(user)

    @staticmethod
    async def deactivate_user(user_id: str) -> bool:
        """deactivate user account."""
        async with get_session() as session:
            repo = UserRepository(session)
            result = await repo.deactivate(user_id)
            if result:
                logger.info(f"deactivated user {user_id}")
            return result

    async def assign_role(self, user_id: str, role: str) -> dict | None:
        """assign role to user."""
        async with get_session() as session:
            repo = UserRepository(session)
            user = await repo.update(user_id, role=role)
            if user is None:
                return None
            logger.info(f"assigned role {role} to user {user_id}")
            return self._user_to_dict(user)

    async def list_users(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """list all users with pagination."""
        async with get_session() as session:
            repo = UserRepository(session)
            users = await repo.list(limit=limit, offset=offset)
            return [self._user_to_dict(user) for user in users]

    # api key management

    async def create_api_key(
        self,
        user_id: str,
        name: str = "Default API Key",
        scopes: list[str] | None = None,
    ) -> tuple[dict, str]:
        """create new api key for user. returns (api_key_dict, plain_key)."""
        async with get_session() as session:
            user_repo = UserRepository(session)
            user = await user_repo.get(user_id)

            if not user:
                raise ValueError(f"user {user_id} not found")

            api_key_repo = APIKeyRepository(session)
            existing_keys = await api_key_repo.list_by_user(user_id)

            plan_limits = PLAN_LIMITS.get(user.plan_tier, PLAN_LIMITS["free"])
            max_keys = plan_limits["max_api_keys"]
            if max_keys != -1 and len(existing_keys) >= max_keys:
                raise ValueError(f"api key limit reached for {user.plan_tier} plan")

            plain_key = self._api_key_prefix + secrets.token_urlsafe(32)
            key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

            if scopes is None:
                scopes = ROLE_SCOPES.get(user.role, [])

            rate_limit = plan_limits["rate_limit_per_minute"]

            api_key = await api_key_repo.create(
                user_id=user_id,
                key_hash=key_hash,
                name=name,
                rate_limit_per_minute=rate_limit,
                scopes=scopes,
            )

            logger.info(f"created api key {api_key.id} for user {user_id}")
            return self._api_key_to_dict(api_key), plain_key

    async def get_api_key(self, api_key_id: str) -> dict | None:
        """get api key by id."""
        async with get_session() as session:
            repo = APIKeyRepository(session)
            api_key = await repo.get(api_key_id)
            if api_key is None:
                return None
            return self._api_key_to_dict(api_key)

    async def validate_api_key(self, plain_key: str) -> dict | None:
        """validate api key and return api key dict if valid."""
        if not plain_key.startswith(self._api_key_prefix):
            return None

        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        async with get_session() as session:
            repo = APIKeyRepository(session)
            api_key = await repo.get_by_hash(key_hash)

            if api_key and api_key.is_active:
                await repo.update_last_used(api_key.id)
                return self._api_key_to_dict(api_key)

        return None

    async def list_user_api_keys(self, user_id: str) -> list[dict]:
        """list all api keys for user."""
        async with get_session() as session:
            repo = APIKeyRepository(session)
            api_keys = await repo.list_by_user(user_id)
            return [self._api_key_to_dict(k) for k in api_keys]

    @staticmethod
    async def deactivate_api_key(api_key_id: str, user_id: str) -> bool:
        """deactivate api key if it belongs to user."""
        async with get_session() as session:
            repo = APIKeyRepository(session)
            api_key = await repo.get(api_key_id)

            if not api_key or api_key.user_id != user_id:
                return False

            result = await repo.deactivate(api_key_id)
            if result:
                logger.info(f"deactivated api key {api_key_id}")
            return result

    async def get_api_key_usage(self, api_key_id: str) -> dict:
        """get current usage stats for api key."""
        current_minute = int(time.time() // 60)
        rate_key = f"rate_limit:{api_key_id}:{current_minute}"

        current_count = await self._redis.client.get(rate_key)
        current_count = int(current_count) if current_count else 0

        async with get_session() as session:
            repo = APIKeyRepository(session)
            api_key = await repo.get(api_key_id)
            rate_limit = api_key.rate_limit_per_minute if api_key else 1000

        return {
            "current_minute_usage": current_count,
            "rate_limit_per_minute": rate_limit,
        }

    # rate limiting

    async def check_rate_limit(self, api_key_id: str) -> tuple[bool, int, int]:
        """
        check rate limit for api key.
        returns (allowed, remaining, limit).
        """
        async with get_session() as session:
            repo = APIKeyRepository(session)
            api_key = await repo.get(api_key_id)

            if not api_key:
                return False, 0, 0

            # check user rate limit first
            user_id = api_key.user_id
            user_allowed, _, _ = await self.check_user_rate_limit(user_id)
            if not user_allowed:
                return False, 0, api_key.rate_limit_per_minute

            current_minute = int(time.time() // 60)
            rate_key = f"rate_limit:{api_key_id}:{current_minute}"

            current_count = await self._redis.client.get(rate_key)
            current_count = int(current_count) if current_count else 0

            if current_count >= api_key.rate_limit_per_minute:
                return False, 0, api_key.rate_limit_per_minute

            await self._redis.client.incr(rate_key)
            await self._redis.client.expire(rate_key, 120)

            remaining = api_key.rate_limit_per_minute - current_count - 1
            return True, remaining, api_key.rate_limit_per_minute

    async def check_user_rate_limit(self, user_id: str) -> tuple[bool, int, int]:
        """
        check rate limit for user.
        returns (allowed, remaining, limit).
        """
        async with get_session() as session:
            repo = UserRepository(session)
            user = await repo.get(user_id)

            if not user:
                return False, 0, 0

            plan_limits = PLAN_LIMITS.get(user.plan_tier, PLAN_LIMITS["free"])
            rate_limit = plan_limits["rate_limit_per_minute"]

            current_minute = int(time.time() // 60)
            rate_key = f"rate_limit:user:{user_id}:{current_minute}"

            current_count = await self._redis.client.get(rate_key)
            current_count = int(current_count) if current_count else 0

            if current_count >= rate_limit:
                return False, 0, rate_limit

            await self._redis.client.incr(rate_key)
            await self._redis.client.expire(rate_key, 120)

            remaining = rate_limit - current_count - 1
            return True, remaining, rate_limit

    # permissions
    @staticmethod
    async def check_user_permission(user_id: str, required_scope: str) -> bool:
        """check if user has required scope based on role."""
        async with get_session() as session:
            repo = UserRepository(session)
            user = await repo.get(user_id)

            if not user:
                return False

            user_scopes = ROLE_SCOPES.get(user.role, [])
            return required_scope in user_scopes or "system:admin" in user_scopes

    @staticmethod
    async def check_api_key_scope(api_key_id: str, required_scope: str) -> bool:
        """check if api key has required scope."""
        async with get_session() as session:
            repo = APIKeyRepository(session)
            api_key = await repo.get(api_key_id)

            if not api_key:
                return False

            return required_scope in api_key.scopes or "system:admin" in api_key.scopes

    @staticmethod
    def get_scopes_for_role(role: str) -> list[str]:
        """get scopes for a given role."""
        return ROLE_SCOPES.get(role, [])

    # conversion helpers

    @staticmethod
    def _user_to_dict(user) -> dict:
        return {
            "id": user.id,
            "tenant_id": user.tenant_id,
            "email": user.email,
            "plan_tier": user.plan_tier,
            "role": user.role,
            "is_active": user.is_active,
            "created_at": user.created_at.timestamp() if user.created_at else None,
            "updated_at": user.updated_at.timestamp() if user.updated_at else None,
        }

    @staticmethod
    def _api_key_to_dict(api_key) -> dict:
        return {
            "id": api_key.id,
            "user_id": api_key.user_id,
            "name": api_key.name,
            "rate_limit_per_minute": api_key.rate_limit_per_minute,
            "scopes": api_key.scopes,
            "is_active": api_key.is_active,
            "created_at": (
                api_key.created_at.timestamp() if api_key.created_at else None
            ),
            "last_used_at": (
                api_key.last_used_at.timestamp() if api_key.last_used_at else None
            ),
        }
