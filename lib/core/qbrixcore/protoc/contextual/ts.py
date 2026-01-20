from typing import ClassVar, Union

import numpy as np
from pydantic import Field, model_validator

from qbrixcore.param.var import ArrayParam
from qbrixcore.param.state import BaseParamState
from qbrixcore.protoc.base import BaseProtocol
from qbrixcore.context import Context


class LinTSParamState(BaseParamState):
    """Parameter state for Linear Thompson Sampling protocol."""

    dim: int = Field(..., gt=0)
    v: float = Field(default=1.0, gt=0.0)
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


class LinTSProtocol(BaseProtocol):
    """
    Linear Thompson Sampling protocol for contextual multi-armed bandit.

    Uses Bayesian linear regression with Gaussian priors to model the
    reward function and samples from the posterior to select arms.
    """

    name: ClassVar[str] = "LinTSProtocol"
    param_state_cls: type[BaseParamState] = LinTSParamState

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
    def _sample_theta(ps: LinTSParamState, arm: int) -> np.ndarray:
        """Sample parameter vector from posterior distribution for an arm."""
        try:
            b_inv = np.linalg.inv(ps.d[arm])
            mu = np.dot(b_inv, ps.r[arm]).flatten()
            cov = (ps.v**2) * b_inv
            cov = (cov + cov.T) / 2
            theta_sample = np.random.multivariate_normal(mu, cov)
            return theta_sample.reshape(-1, 1)
        except np.linalg.LinAlgError:
            try:
                b_inv = np.linalg.pinv(ps.d[arm])
                mu = np.dot(b_inv, ps.r[arm])
                return mu
            except Exception:  # noqa
                return np.zeros((ps.dim, 1), dtype=np.float64)

    @staticmethod
    def select(ps: LinTSParamState, context: Context) -> int:
        """Arm selection using Linear Thompson Sampling."""
        x = LinTSProtocol._reshape_context_vector(context)
        pred = []
        for arm in range(ps.num_arms):
            theta_sample = LinTSProtocol._sample_theta(ps, arm)
            expected_reward = np.dot(theta_sample.T, x).item()
            pred.append(expected_reward)
        return int(np.argmax(pred))

    @classmethod
    def train(
        cls,
        ps: LinTSParamState,
        context: Context,
        choice: int,
        reward: Union[int, float, np.float64],
    ) -> LinTSParamState:
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
