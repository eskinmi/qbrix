from abc import ABC, abstractmethod

from .state import BaseParamState


class BaseParamBackend(ABC):

    @abstractmethod
    def get(self, experiment_id: str):
        pass

    @abstractmethod
    def set(self, experiment_id: str, params: BaseParamState):
        pass


class InMemoryParamBackend(BaseParamBackend):

    def __init__(self):
        self.store = dict()

    def get(self, experiment_id: str):  # todo: need cache
        return self.store.get(experiment_id)

    def set(self, experiment_id: str, params: BaseParamState):
        self.store.update({experiment_id: params})
