import math
from typing import ClassVar, Union

import numpy as np
from pydantic import Field, model_validator

from qbrixcore.param.var import ArrayParam
from qbrixcore.param.state import BaseParamState
from qbrixcore.policy.base import BasePolicy
from qbrixcore.context import Context


class UCB1TunedParamState(BaseParamState):
    """Parameter state for UCB1-Tuned policy."""

    alpha: float = Field(default=2.0, gt=0.0)
    mu: ArrayParam | None = None
    T: ArrayParam | None = None
    rsq: ArrayParam | None = None
    round: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def set_defaults(self):
        if self.mu is None:
            self.mu = np.zeros(self.num_arms, dtype=np.float64)
        if self.T is None:
            self.T = np.zeros(self.num_arms, dtype=np.int64)
        if self.rsq is None:
            self.rsq = np.zeros(self.num_arms, dtype=np.float64)
        return self


class UCB1TunedPolicy(BasePolicy):
    """
    UCB1-Tuned policy for multi-armed bandit.

    Uses variance estimates to compute tighter confidence bounds than UCB1.
    """

    name: ClassVar[str] = "UCB1TunedPolicy"
    param_state_cls: type[BaseParamState] = UCB1TunedParamState

    @staticmethod
    def _arm_var_upper_bound(ps: UCB1TunedParamState, arm: int) -> float:
        """Calculate arm variance upper bound."""
        if ps.T[arm] == 0:
            return float("inf")
        sigma = ps.rsq[arm] / ps.T[arm] - ps.mu[arm] ** 2
        delta = math.sqrt(ps.alpha * math.log(ps.round + 1) / ps.T[arm])
        return float(sigma + delta)

    @staticmethod
    def _upper_bound(ps: UCB1TunedParamState, arm: int) -> float:
        """Calculate upper confidence bound."""
        if ps.T[arm] == 0:
            return float("inf")
        sigma_bound = min(0.25, UCB1TunedPolicy._arm_var_upper_bound(ps, arm))
        return float(
            ps.mu[arm] + math.sqrt(sigma_bound * math.log(ps.round + 1) / ps.T[arm])
        )

    @staticmethod
    def select(ps: UCB1TunedParamState, context: Context) -> int:
        """Arm selection using UCB1-Tuned."""
        # Note: round increment happens in train, use current round for selection
        upper_bounds = [
            UCB1TunedPolicy._upper_bound(ps, i) for i in range(ps.num_arms)
        ]
        return int(np.argmax(upper_bounds))

    @classmethod
    def train(
        cls,
        ps: UCB1TunedParamState,
        context: Context,
        choice: int,
        reward: Union[int, float, np.float64],
    ) -> UCB1TunedParamState:
        """Update state with observed reward."""
        new_T = ps.T.copy()
        new_mu = ps.mu.copy()
        new_rsq = ps.rsq.copy()

        new_T[choice] += 1
        new_rsq[choice] += reward**2
        prev_mu = ps.mu[choice]
        new_mu[choice] += (reward - prev_mu) / new_T[choice]

        return ps.model_copy(
            update={
                "T": new_T,
                "mu": new_mu,
                "rsq": new_rsq,
                "round": ps.round + 1,
            }
        )


class KLUCBParamState(BaseParamState):
    """Parameter state for KL-UCB policy."""

    c: float = Field(default=0.0, ge=0.0)
    S: ArrayParam | None = None
    N: ArrayParam | None = None
    round: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def set_defaults(self):
        if self.S is None:
            self.S = np.zeros(self.num_arms, dtype=np.float64)
        if self.N is None:
            self.N = np.zeros(self.num_arms, dtype=np.int64)
        return self


class KLUCBPolicy(BasePolicy):
    """
    KL-UCB (Kullback-Leibler Upper Confidence Bound) policy.

    Based on "The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond"
    by Garivier & Cappe (2011).

    Uses KL-divergence to compute tighter confidence bounds than standard UCB,
    achieving the Lai-Robbins lower bound for Bernoulli rewards.
    """

    name: ClassVar[str] = "KLUCBPolicy"
    param_state_cls: type[BaseParamState] = KLUCBParamState

    tolerance: float = 1e-6
    max_iterations: int = 50

    @staticmethod
    def _kl_bernoulli(p: float, q: float) -> float:
        """Compute KL divergence between Bernoulli(p) and Bernoulli(q)."""
        p = np.clip(p, 0.0, 1.0)
        q = np.clip(q, 0.0, 1.0)

        if p == 0.0:
            if q == 1.0:
                return float("inf")
            return -math.log(1.0 - q)

        if p == 1.0:
            if q == 0.0:
                return float("inf")
            return -math.log(q)

        if q == 0.0 or q == 1.0:
            return float("inf")

        return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))

    def _compute_ucb(self, ps: KLUCBParamState, arm: int, t: int) -> float:
        """Compute KL-UCB upper confidence bound for an arm."""
        if ps.N[arm] == 0:
            return float("inf")

        p_hat = ps.S[arm] / ps.N[arm]
        n = ps.N[arm]

        if t <= 1:
            threshold = 0.0
        else:
            log_t = math.log(t)
            log_log_t = math.log(log_t) if log_t > 1.0 else 0.0
            threshold = (log_t + ps.c * log_log_t) / n

        if threshold < 1e-10:
            return p_hat

        left, right = p_hat, 1.0

        if self._kl_bernoulli(p_hat, right) <= threshold:
            return right

        for _ in range(self.max_iterations):
            mid = (left + right) / 2.0
            kl_div = self._kl_bernoulli(p_hat, mid)

            if abs(kl_div - threshold) < self.tolerance:
                return mid

            if kl_div < threshold:
                left = mid
            else:
                right = mid

            if abs(right - left) < self.tolerance:
                break

        return (left + right) / 2.0

    def select(self, ps: KLUCBParamState, context: Context) -> int:
        """Arm selection using KL-UCB."""
        t = ps.round + 1
        ucb_values = [self._compute_ucb(ps, i, t) for i in range(ps.num_arms)]
        return int(np.argmax(ucb_values))

    @classmethod
    def train(
        cls,
        ps: KLUCBParamState,
        context: Context,
        choice: int,
        reward: Union[int, float, np.float64],
    ) -> KLUCBParamState:
        """Update state with observed reward."""
        new_N = ps.N.copy()
        new_S = ps.S.copy()

        reward = np.clip(reward, 0.0, 1.0)
        new_N[choice] += 1
        new_S[choice] += reward

        return ps.model_copy(
            update={
                "N": new_N,
                "S": new_S,
                "round": ps.round + 1,
            }
        )


class KLUCBPlusPolicy(KLUCBPolicy):
    """
    KL-UCB+ variant using log(t/N[a]) instead of log(t) in exploration bonus.

    This variant can provide better empirical performance. Inspired by MOSS and DMED+.
    """

    name: ClassVar[str] = "KLUCBPlusPolicy"
    param_state_cls: type[BaseParamState] = KLUCBParamState

    def _compute_ucb(self, ps: KLUCBParamState, arm: int, t: int) -> float:
        """Compute KL-UCB+ upper confidence bound using log(t/N[arm])."""
        if ps.N[arm] == 0:
            return float("inf")

        p_hat = ps.S[arm] / ps.N[arm]
        n = ps.N[arm]

        ratio = max(t / n, 1.0)
        log_ratio = math.log(ratio)

        if log_ratio <= 0:
            return p_hat

        log_log_ratio = math.log(log_ratio) if log_ratio > 1.0 else 0.0
        threshold = (log_ratio + ps.c * log_log_ratio) / n

        if threshold < 1e-10:
            return p_hat

        left, right = p_hat, 1.0

        if self._kl_bernoulli(p_hat, right) <= threshold:
            return right

        for _ in range(self.max_iterations):
            mid = (left + right) / 2.0
            kl_div = self._kl_bernoulli(p_hat, mid)

            if abs(kl_div - threshold) < self.tolerance:
                return mid

            if kl_div < threshold:
                left = mid
            else:
                right = mid

            if abs(right - left) < self.tolerance:
                break

        return (left + right) / 2.0
