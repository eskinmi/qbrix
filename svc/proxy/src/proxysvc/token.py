"""
selection token utilities for encoding/decoding selection context.

uses HMAC-SHA256 for signing and base64 for encoding. the token contains
all selection context data, eliminating the need for server-side storage.
"""

from __future__ import annotations

import base64
import hmac
import json
import time
from dataclasses import dataclass


class TokenError(Exception):
    """raised when token validation fails."""

    pass


class TokenExpiredError(TokenError):
    """raised when token has expired."""

    pass


class TokenInvalidError(TokenError):
    """raised when token signature is invalid."""

    pass


@dataclass
class SelectionEntry:
    """decoded selection data from token."""

    experiment_id: str
    arm_index: int
    context_id: str
    context_vector: list[float]
    context_metadata: dict
    timestamp_ms: int


class SelectionToken:

    @staticmethod
    def encode(
        secret: bytes,
        experiment_id: str,
        arm_index: int,
        context_id: str,
        context_vector: list[float],
        context_metadata: dict,
    ) -> str:
        """
        create a signed token containing selection data.

        args:
            secret: hmac signing key
            experiment_id: experiment identifier
            arm_index: selected arm index
            context_id: context identifier
            context_vector: context feature vector
            context_metadata: context metadata dict

        returns:
            base64-encoded signed token
        """
        payload = {
            "exp_id": experiment_id,
            "arm_idx": arm_index,
            "ctx_id": context_id,
            "ctx_vec": context_vector,
            "ctx_meta": context_metadata,
            "ts": int(time.time() * 1000),
        }
        data = json.dumps(payload, separators=(",", ":")).encode()
        sig = hmac.new(secret, data, "sha256").digest()[:16]
        return base64.urlsafe_b64encode(data + sig).decode()

    @staticmethod
    def decode(
        secret: bytes,
        token: str,
        max_age_ms: int | None = None,
    ) -> SelectionEntry:
        """
        verify and decode a selection token.

        args:
            secret: hmac signing key (must match key used to create token)
            token: base64-encoded signed token
            max_age_ms: maximum token age in milliseconds (None = no expiry check)

        returns:
            SelectionData with decoded selection context

        raises:
            TokenInvalidError: if signature verification fails
            TokenExpiredError: if token has expired
        """
        try:
            raw = base64.urlsafe_b64decode(token)
        except Exception as e:
            raise TokenInvalidError(f"failed to decode token: {e}")

        if len(raw) < 17:
            raise TokenInvalidError("token too short")

        data, sig = raw[:-16], raw[-16:]
        expected_sig = hmac.new(secret, data, "sha256").digest()[:16]

        if not hmac.compare_digest(sig, expected_sig):
            raise TokenInvalidError("invalid token signature")

        try:
            payload = json.loads(data)
        except json.JSONDecodeError as e:
            raise TokenInvalidError(f"failed to parse token payload: {e}")

        if max_age_ms is not None:
            age_ms = int(time.time() * 1000) - payload["ts"]
            if age_ms > max_age_ms:
                raise TokenExpiredError(f"token expired ({age_ms}ms > {max_age_ms}ms)")

        return SelectionEntry(
            experiment_id=payload["exp_id"],
            arm_index=payload["arm_idx"],
            context_id=payload["ctx_id"],
            context_vector=payload["ctx_vec"],
            context_metadata=payload["ctx_meta"],
            timestamp_ms=payload["ts"],
        )
