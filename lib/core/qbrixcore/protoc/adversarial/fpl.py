from typing import ClassVar, Union

import numpy as np
from pydantic import Field, model_validator

from qbrixcore.param.var import ArrayParam
from qbrixcore.param.state import BaseParamState
from qbrixcore.protoc.base import BaseProtocol
from qbrixcore.context import Context


class FPLParamState(BaseParamState):
    """Parameter state for FPL (Follow the Perturbed Leader) protocol."""

    eta: float = Field(default=5.0, gt=0.0)
    r: ArrayParam | None = None

    @model_validator(mode="after")
    def set_defaults(self):
        if self.r is None:
            self.r = np.zeros(self.num_arms, dtype=np.float64)
        return self


class FPLProtocol(BaseProtocol):
    """
    FPL (Follow the Perturbed Leader) protocol for adversarial multi-armed bandit.

    FPL works by:
    1. Maintaining cumulative rewards for each arm
    2. Adding random noise to cumulative rewards during selection
    3. Selecting the arm with the highest perturbed cumulative reward
    4. Updating only the actual reward (no noise in updates)

    The eta parameter controls the noise scale (exploration).
    """

    name: ClassVar[str] = "FPLProtocol"
    param_state_cls: type[BaseParamState] = FPLParamState

    @staticmethod
    def select(ps: FPLParamState, context: Context) -> int:
        """Arm selection using FPL with exponential perturbation."""
        noise = np.random.exponential(ps.eta, size=ps.num_arms)
        perturbed_rewards = ps.r + noise
        return int(np.argmax(perturbed_rewards))

    @classmethod
    def train(
        cls,
        ps: FPLParamState,
        context: Context,
        choice: int,
        reward: Union[int, float, np.float64],
    ) -> FPLParamState:
        """Update cumulative rewards (no noise in updates)."""
        new_r = ps.r.copy()
        new_r[choice] += reward
        return ps.model_copy(update={"r": new_r})
