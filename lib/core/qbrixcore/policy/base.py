from abc import ABC, abstractmethod

from qbrixcore.param.state import BaseParamState


class BasePolicy(ABC):
    """
    Base class for all criteria used in bandit algorithms.
    """

    param_state_cls: type[BaseParamState]

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def select(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def train(cls, *args, **kwargs):
        pass

    @classmethod
    def init_params(cls, **params) -> BaseParamState:
        return cls.param_state_cls(**params)
