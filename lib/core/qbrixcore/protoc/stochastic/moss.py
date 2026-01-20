import math
from typing import ClassVar, Union

import numpy as np
from pydantic import Field, model_validator

from qbrixcore.param.var import ArrayParam
from qbrixcore.param.state import BaseParamState
from qbrixcore.protoc.base import BaseProtocol
from qbrixcore.context import Context


class MOSSParamState(BaseParamState):
    """Parameter state for MOSS (Minimax Optimal Strategy in the Stochastic case) protocol."""

    horizon: int = Field(..., gt=0)
    mu: ArrayParam | None = None
    T: ArrayParam | None = None
    round: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def set_defaults(self):
        if self.mu is None:
            self.mu = np.zeros(self.num_arms, dtype=np.float64)
        if self.T is None:
            self.T = np.zeros(self.num_arms, dtype=np.int64)
        return self


class MOSSProtocol(BaseProtocol):
    """
    MOSS (Minimax Optimal Strategy in the Stochastic case) protocol.

    Based on "Minimax Policies for Adversarial and Stochastic Bandits"
    by Audibert & Bubeck (2009).

    MOSS achieves minimax optimal worst-case regret of O(sqrt(nK)) while maintaining
    instance-dependent logarithmic regret.

    The MOSS index for arm i at time t is:
        mu_i + sqrt(max(log(n/(K*T_i)), 0) / T_i)

    Regret bounds:
        - Worst-case (minimax): R_n <= 49*sqrt(nK)
        - Instance-dependent: R_n <= 23K * sum_i max(log(n*delta_i^2/K), 1) / delta_i
    """

    name: ClassVar[str] = "MOSSProtocol"
    param_state_cls: type[BaseParamState] = MOSSParamState

    @staticmethod
    def _moss_index(ps: MOSSParamState, arm: int) -> float:
        """
        Calculate MOSS index (upper confidence bound) for an arm.

        Formula: mu_i + sqrt(max(log(n/(K*T_i)), 0) / T_i)
        """
        if ps.T[arm] == 0:
            return float("inf")

        n, k, t_i, mu_i = ps.horizon, ps.num_arms, ps.T[arm], ps.mu[arm]
        log_term = math.log(n / (k * t_i)) if n > k * t_i else 0.0
        exploration_bonus = math.sqrt(max(log_term, 0.0) / t_i)

        return float(mu_i + exploration_bonus)

    @staticmethod
    def select(ps: MOSSParamState, context: Context) -> int:
        """
        Arm selection using MOSS index.

        Selects the arm with the highest MOSS index:
            argmax_i [mu_i + sqrt(max(log(n/(K*T_i)), 0) / T_i)]
        """
        moss_indices = [MOSSProtocol._moss_index(ps, i) for i in range(ps.num_arms)]
        return int(np.argmax(moss_indices))

    @classmethod
    def train(
        cls,
        ps: MOSSParamState,
        context: Context,
        choice: int,
        reward: Union[int, float, np.float64],
    ) -> MOSSParamState:
        """
        Update state with observed reward.

        Updates empirical mean using incremental average formula:
            mu_new = mu_old + (reward - mu_old) / T_i
        """
        new_T = ps.T.copy()
        new_mu = ps.mu.copy()
        new_T[choice] += 1
        new_mu[choice] += (reward - ps.mu[choice]) / new_T[choice]

        return ps.model_copy(
            update={
                "T": new_T,
                "mu": new_mu,
                "round": ps.round + 1,
            }
        )


class MOSSAnyTimeParamState(BaseParamState):
    """Parameter state for Anytime MOSS protocol (no horizon required)."""

    mu: ArrayParam | None = None
    T: ArrayParam | None = None
    round: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def set_defaults(self):
        if self.mu is None:
            self.mu = np.zeros(self.num_arms, dtype=np.float64)
        if self.T is None:
            self.T = np.zeros(self.num_arms, dtype=np.int64)
        return self


class MOSSAnyTimeProtocol(BaseProtocol):
    """
    Anytime MOSS protocol that doesn't require horizon knowledge.

    Uses current round number as a proxy for the horizon in the MOSS index:
        mu_i + sqrt(max(log(t/(K*T_i)), 0) / T_i)

    This allows the algorithm to be used when the time horizon is not known
    in advance, at the cost of slightly worse constants in the regret bound.
    """

    name: ClassVar[str] = "MOSSAnyTimeProtocol"
    param_state_cls: type[BaseParamState] = MOSSAnyTimeParamState

    @staticmethod
    def _moss_anytime_index(ps: MOSSAnyTimeParamState, arm: int) -> float:
        """
        Calculate anytime MOSS index using current round as horizon.

        Formula: mu_i + sqrt(max(log(t/(K*T_i)), 0) / T_i)
        """
        if ps.T[arm] == 0:
            return float("inf")

        t = max(ps.round, 1)
        K = ps.num_arms
        T_i = ps.T[arm]
        mu_i = ps.mu[arm]

        log_term = math.log(t / (K * T_i)) if t > K * T_i else 0.0
        exploration_bonus = math.sqrt(max(log_term, 0.0) / T_i)

        return float(mu_i + exploration_bonus)

    @staticmethod
    def select(ps: MOSSAnyTimeParamState, context: Context) -> int:
        """Arm selection using anytime MOSS index."""
        moss_indices = [
            MOSSAnyTimeProtocol._moss_anytime_index(ps, i) for i in range(ps.num_arms)
        ]
        return int(np.argmax(moss_indices))

    @classmethod
    def train(
        cls,
        ps: MOSSAnyTimeParamState,
        context: Context,
        choice: int,
        reward: Union[int, float, np.float64],
    ) -> MOSSAnyTimeParamState:
        """Update state with observed reward."""
        new_T = ps.T.copy()
        new_mu = ps.mu.copy()

        new_T[choice] += 1
        new_mu[choice] += (reward - ps.mu[choice]) / new_T[choice]

        return ps.model_copy(
            update={
                "T": new_T,
                "mu": new_mu,
                "round": ps.round + 1,
            }
        )
