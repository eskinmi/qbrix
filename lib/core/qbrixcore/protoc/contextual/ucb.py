from typing import ClassVar, Union

import numpy as np
from pydantic import Field, model_validator

from qbrixcore.param.var import ArrayParam
from qbrixcore.param.state import BaseParamState
from qbrixcore.protoc.base import BaseProtocol
from qbrixcore.context import Context


class LinUCBParamState(BaseParamState):
    """Parameter state for Linear UCB protocol."""

    dim: int = Field(..., gt=0)
    alpha: float = Field(default=1.5, gt=0.0)
    d: ArrayParam | None = None  # design matrices (num_arms, dim, dim)
    r: ArrayParam | None = None  # reward weighted context sum (num_arms, dim, 1)

    @model_validator(mode="after")
    def set_defaults(self):
        if self.d is None:
            self.d = np.array(
                [np.identity(self.dim, dtype=np.float64) for _ in range(self.num_arms)]
            )
        if self.r is None:
            self.r = np.zeros((self.num_arms, self.dim, 1), dtype=np.float64)
        return self


class LinUCBProtocol(BaseProtocol):
    """
    Linear UCB protocol for contextual multi-armed bandit.

    Uses ridge regression to estimate reward parameters and adds
    confidence bounds based on the design matrix inverse.
    """

    name: ClassVar[str] = "LinUCBProtocol"
    param_state_cls: type[BaseParamState] = LinUCBParamState

    @staticmethod
    def _reshape_context_vector(context: Context) -> np.ndarray:
        """Reshape context vector to column vector."""
        context_vector = context.vector
        if isinstance(context_vector, list):
            context_vector = np.array(context_vector, dtype=np.float64)
        if context_vector.ndim == 1:
            context_vector = context_vector.reshape(-1, 1)
        return context_vector

    @staticmethod
    def _arm_upper_bound(ps: LinUCBParamState, arm: int, context: np.ndarray) -> float:
        """Calculate upper confidence bound for an arm."""
        try:
            a_inv = np.linalg.inv(ps.d[arm])
            theta = np.dot(a_inv, ps.r[arm])
            mean_estimate = np.dot(theta.T, context)
            confidence_bound = ps.alpha * np.sqrt(
                np.dot(context.T, np.dot(a_inv, context))
            )
            return float((mean_estimate + confidence_bound).item())
        except np.linalg.LinAlgError:
            return float("inf")

    @staticmethod
    def select(ps: LinUCBParamState, context: Context) -> int:
        """Arm selection using Linear UCB."""
        x = LinUCBProtocol._reshape_context_vector(context)
        upper_bounds = [
            LinUCBProtocol._arm_upper_bound(ps, i, x) for i in range(ps.num_arms)
        ]
        return int(np.argmax(upper_bounds))

    @classmethod
    def train(
        cls,
        ps: LinUCBParamState,
        context: Context,
        choice: int,
        reward: Union[int, float, np.float64],
    ) -> LinUCBParamState:
        """Update state with observed reward."""
        x = cls._reshape_context_vector(context)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        new_d = ps.d.copy()
        new_r = ps.r.copy()

        new_d[choice] = ps.d[choice] + np.dot(x, x.T)
        new_r[choice] = ps.r[choice] + reward * x

        return ps.model_copy(
            update={
                "d": new_d,
                "r": new_r,
            }
        )
