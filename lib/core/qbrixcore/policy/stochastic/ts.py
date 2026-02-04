from typing import ClassVar, Union

import numpy as np
from pydantic import Field, model_validator

from qbrixcore.param.var import ArrayParam
from qbrixcore.param.state import BaseParamState
from qbrixcore.policy.base import BasePolicy
from qbrixcore.context import Context


class BetaTSParamState(BaseParamState):
    """Parameter state for Beta-Bernoulli Thompson Sampling policy."""

    alpha_prior: float = Field(default=1.0, gt=0.0)
    beta_prior: float = Field(default=1.0, gt=0.0)
    alpha: ArrayParam | None = None
    beta: ArrayParam | None = None
    T: ArrayParam | None = None

    @model_validator(mode="after")
    def set_defaults(self):
        if self.alpha is None:
            self.alpha = np.full(self.num_arms, self.alpha_prior, dtype=np.float64)
        if self.beta is None:
            self.beta = np.full(self.num_arms, self.beta_prior, dtype=np.float64)
        if self.T is None:
            self.T = np.zeros(self.num_arms, dtype=np.int64)
        return self


class BetaTSPolicy(BasePolicy):
    """
    Beta-Bernoulli Thompson Sampling policy for binary rewards.

    Uses Beta distributions as conjugate priors for Bernoulli likelihoods.
    Best suited for binary rewards (0/1) or rewards that can be interpreted
    as success rates.
    """

    name: ClassVar[str] = "BetaTSPolicy"
    param_state_cls: type[BaseParamState] = BetaTSParamState

    @staticmethod
    def select(ps: BetaTSParamState, context: Context) -> int:
        """Arm selection using Thompson Sampling."""
        samples = np.random.beta(ps.alpha, ps.beta, size=ps.num_arms)
        return int(np.argmax(samples))

    @classmethod
    def train(
        cls,
        ps: BetaTSParamState,
        context: Context,
        choice: int,
        reward: Union[int, float, np.float64],
    ) -> BetaTSParamState:
        """
        Update state with observed reward using Beta-Bernoulli conjugacy.

        Converts reward to binary: 1 if reward > 0.5, else 0.
        """
        new_alpha = ps.alpha.copy()
        new_beta = ps.beta.copy()
        new_T = ps.T.copy()

        # convert reward to binary
        if reward not in [0, 1]:
            binary_reward = 1 if reward > 0.5 else 0
        else:
            binary_reward = int(reward)

        new_T[choice] += 1
        if binary_reward == 1:
            new_alpha[choice] += 1
        else:
            new_beta[choice] += 1

        return ps.model_copy(
            update={
                "alpha": new_alpha,
                "beta": new_beta,
                "T": new_T,
            }
        )


class GaussianTSParamState(BaseParamState):
    """Parameter state for Gaussian Thompson Sampling policy."""

    prior_mean: float = Field(default=0.0)
    prior_precision: float = Field(default=1.0, gt=0.0)
    noise_precision: float = Field(default=1.0, gt=0.0)
    posterior_mean: ArrayParam | None = None
    posterior_precision: ArrayParam | None = None
    T: ArrayParam | None = None

    @model_validator(mode="after")
    def set_defaults(self):
        if self.posterior_mean is None:
            self.posterior_mean = np.full(
                self.num_arms, self.prior_mean, dtype=np.float64
            )
        if self.posterior_precision is None:
            self.posterior_precision = np.full(
                self.num_arms, self.prior_precision, dtype=np.float64
            )
        if self.T is None:
            self.T = np.zeros(self.num_arms, dtype=np.int64)
        return self


class GaussianTSPolicy(BasePolicy):
    """
    Gaussian Thompson Sampling policy for continuous rewards.

    Uses Gaussian distributions with conjugate Gaussian priors.
    Assumes rewards are normally distributed and updates both mean and precision.
    """

    name: ClassVar[str] = "GaussianTSPolicy"
    param_state_cls: type[BaseParamState] = GaussianTSParamState

    @staticmethod
    def select(ps: GaussianTSParamState, context: Context) -> int:
        """Arm selection using Gaussian Thompson Sampling."""
        samples = [
            np.random.normal(
                ps.posterior_mean[i], 1.0 / np.sqrt(ps.posterior_precision[i])
            )
            for i in range(ps.num_arms)
        ]
        return int(np.argmax(samples))

    @classmethod
    def train(
        cls,
        ps: GaussianTSParamState,
        context: Context,
        choice: int,
        reward: Union[int, float, np.float64],
    ) -> GaussianTSParamState:
        """Update state with observed reward using Gaussian-Gaussian conjugacy."""
        new_posterior_mean = ps.posterior_mean.copy()
        new_posterior_precision = ps.posterior_precision.copy()
        new_T = ps.T.copy()

        new_T[choice] += 1

        prev_precision = ps.posterior_precision[choice]
        prev_mean = ps.posterior_mean[choice]

        new_posterior_precision[choice] = prev_precision + ps.noise_precision
        new_posterior_mean[choice] = (
            prev_precision * prev_mean + ps.noise_precision * reward
        ) / new_posterior_precision[choice]

        return ps.model_copy(
            update={
                "posterior_mean": new_posterior_mean,
                "posterior_precision": new_posterior_precision,
                "T": new_T,
            }
        )
