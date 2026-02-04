from typing import ClassVar, Union

import numpy as np
from pydantic import Field, model_validator

from qbrixcore.param.var import ArrayParam
from qbrixcore.param.state import BaseParamState
from qbrixcore.policy.base import BasePolicy
from qbrixcore.context import Context


class EXP3ParamState(BaseParamState):
    """Parameter state for EXP3 (Exponential-weight) policy."""

    gamma: float = Field(default=0.1, ge=0.0, le=1.0)
    w: ArrayParam | None = None

    @model_validator(mode="after")
    def set_defaults(self):
        if self.w is None:
            self.w = np.ones(self.num_arms, dtype=np.float64)
        return self


class EXP3Policy(BasePolicy):
    """
    EXP3 (Exponential-weight algorithm for Exploration and Exploitation)
    policy for adversarial multi-armed bandit.

    EXP3 maintains weights for each arm and uses importance-weighted
    reward estimates to update them. The gamma parameter controls
    the exploration-exploitation tradeoff.
    """

    name: ClassVar[str] = "EXP3Policy"
    param_state_cls: type[BaseParamState] = EXP3ParamState

    @staticmethod
    def _proba(ps: EXP3ParamState) -> np.ndarray:
        """Compute selection probabilities."""
        return (1.0 - ps.gamma) * (ps.w / ps.w.sum()) + (ps.gamma / ps.num_arms)

    @staticmethod
    def select(ps: EXP3ParamState, context: Context) -> int:
        """Arm selection using EXP3."""
        proba = EXP3Policy._proba(ps)
        return int(np.random.choice(ps.num_arms, p=proba))

    @classmethod
    def train(
        cls,
        ps: EXP3ParamState,
        context: Context,
        choice: int,
        reward: Union[int, float, np.float64],
    ) -> EXP3ParamState:
        """Update weights with importance-weighted reward estimate."""
        proba = cls._proba(ps)

        # importance-weighted reward estimate
        rewards = np.zeros(ps.num_arms)
        rewards[choice] = reward
        estimate = rewards / proba

        # update weights
        new_w = ps.w.copy()
        new_w *= np.exp(estimate * ps.gamma / ps.num_arms)
        new_w /= np.sum(new_w)  # normalize to prevent overflow

        return ps.model_copy(update={"w": new_w})
