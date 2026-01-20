from typing import ClassVar, Union
import random

from pydantic import Field, model_validator
import numpy as np

from qbrixcore.param.var import ArrayParam
from qbrixcore.param.state import BaseParamState
from qbrixcore.protoc.base import BaseProtocol
from qbrixcore.context import Context


class EpsilonParamState(BaseParamState):
    eps: float = Field(..., ge=0.0, le=1.0)
    gamma: float = Field(..., ge=0.0, le=1.0)
    mu: ArrayParam | None = None
    T: ArrayParam | None = None

    @model_validator(mode="after")
    def set_defaults(self):
        if self.mu is None:
            self.mu = np.zeros(self.num_arms, dtype=np.float64)
        if self.T is None:
            self.T = np.zeros(self.num_arms, dtype=np.int64)
        return self


class EpsilonProtocol(BaseProtocol):

    name: ClassVar[str] = "EpsilonProtocol"
    param_state_cls: type[BaseParamState] = EpsilonParamState

    @staticmethod
    def decay(ps: EpsilonParamState):
        """
        Thread-safe epsilon decay using exponential decay.

        Formula: eps_new = eps_old * (1 - gamma)

        Examples:
        - gamma=0.0: No decay (eps stays constant)
        - gamma=0.01: Slow decay (1% reduction per step)
        - gamma=0.1: Medium decay (10% reduction per step)
        - gamma=0.5: Fast decay (50% reduction per step)
        """
        ps.eps *= 1 - ps.gamma
        return ps

    @staticmethod
    def select(ps: EpsilonParamState, context: Context):
        if random.random() > ps.eps:
            return int(np.argmax(ps.mu))
        else:
            return random.choice(range(ps.num_arms))

    @classmethod
    def train(
        cls,
        ps: EpsilonParamState,
        context: Context,
        choice: int,
        reward: Union[int, float, np.float64],
    ) -> EpsilonParamState:

        new_T = ps.T.copy()
        new_mu = ps.mu.copy()

        new_T[choice] += 1
        new_mu[choice] += (reward - ps.mu[choice]) / new_T[choice]
        new_eps = ps.eps * (1 - ps.gamma)

        return ps.model_copy(
            update={
                "T": new_T,
                "mu": new_mu,
                "eps": new_eps,
            }
        )
