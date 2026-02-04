from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

import requests

from .config import settings


@dataclass
class SelectResult:
    request_id: str
    arm_id: str
    arm_index: int
    is_default: bool


class ProxyClient:
    """HTTP client wrapper for load testing."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or settings.proxy_address).rstrip("/")
        self.api_key = api_key or settings.api_key
        self._session: Optional[requests.Session] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update(
                {"X-API-Key": self.api_key}
            )

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def session(self) -> requests.Session:
        if not self._session:
            raise RuntimeError("client not connected, call connect() first")
        return self._session

    # ------------------------------------------------------------------ #
    # endpoints
    # ------------------------------------------------------------------ #

    def health_check(self) -> bool:
        r = self.session.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        return True

    def create_pool(self, name: str, num_arms: int) -> str:
        payload = {
            "name": name,
            "arms": [
                {"name": f"arm-{i}"} for i in range(num_arms)
            ],
        }
        r = self.session.post(
            f"{self.base_url}/api/v1/pools",
            json=payload,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["id"]

    def delete_pool(self, pool_id: str) -> bool:
        r = self.session.delete(
            f"{self.base_url}/api/v1/pools/{pool_id}",
            timeout=10,
        )
        r.raise_for_status()
        return True

    def create_experiment(
        self,
        name: str,
        pool_id: str,
        policy: str = "beta_ts",
        enabled: bool = True,
    ) -> str:
        payload = {
            "name": name,
            "pool_id": pool_id,
            "policy": policy,
            "enabled": enabled,
        }
        r = self.session.post(
            f"{self.base_url}/api/v1/experiments",
            json=payload,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["id"]

    def delete_experiment(self, experiment_id: str) -> bool:
        r = self.session.delete(
            f"{self.base_url}/api/v1/experiments/{experiment_id}",
            timeout=10,
        )
        r.raise_for_status()
        return True

    # ------------------------------------------------------------------ #
    # bandit inference (ASSUMED endpoints)
    # ------------------------------------------------------------------ #

    def select(
        self,
        experiment_id: str,
        context_id: str | None = None,
        context_vector: list[float] | None = None,
        context_metadata: dict[str, str] | None = None,
    ) -> SelectResult:
        """
        ⚠️ Endpoint not present in provided OpenAPI.
        Adjust path/payload to match your actual server.
        """

        payload = {
            "experiment_id": experiment_id,
            "context": {
                "id": context_id or str(uuid.uuid4()),
                "vector": context_vector or [],
                "metadata": context_metadata or {},
            },
        }

        r = self.session.post(
            f"{self.base_url}/api/v1/agent/select",
            json=payload,
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()

        return SelectResult(
            request_id=data["request_id"],
            arm_id=data["arm"]["id"],
            arm_index=data["arm"]["index"],
            is_default=data.get("is_default", False),
        )

    def feedback(self, request_id: str, reward: float) -> bool:

        payload = {
            "request_id": request_id,
            "reward": reward,
        }

        r = self.session.post(
            f"{self.base_url}/api/v1/agent/feedback",
            json=payload,
            timeout=5,
        )
        r.raise_for_status()
        return r.json().get("accepted", True)
