from dataclasses import dataclass, field
from typing import Union, List
import uuid

from qbrixcore import callback
from qbrixcore.pool import Pool
from qbrixcore.protoc.base import BaseProtocol
from qbrixcore.param.backend import BaseParamBackend, InMemoryParamBackend
from qbrixcore.context import Context


@dataclass
class Agent:

    experiment_id: str = field(metadata={'description': 'experiment id the agent belongs to.'})
    pool: Pool = field(metadata={'description': 'pool of arms for the experiment.'})
    protocol: type[BaseProtocol] = field(metadata={'description': 'the protocol used for the experiment.'})
    init_params: dict = field(default_factory=dict)
    param_backend: BaseParamBackend | None = field(default=None, metadata={'description': 'parameter storage backend'})
    id: str = field(default_factory=lambda: str(uuid.uuid4().hex), metadata={'description': 'unique agent id.'})
    callbacks: List[callback.BaseCallback] = field(default_factory=list)


    def __post_init__(self):
        if self.param_backend is None:
            self.param_backend = InMemoryParamBackend()


    def add_callback(self, clb: callback.BaseCallback):
        """Thread-safe callback registration"""
        if not isinstance(clb, callback.BaseCallback):
            raise TypeError("Callback must be an instance of BaseCallback")  # noqa
        self.callbacks.append(clb)

    @callback.register()
    def select(self, context: Context):
        paramstate = self.param_backend.get(experiment_id=self.experiment_id)
        if paramstate is None:
            raise RuntimeError(
                f"param state not found for experiment {self.experiment_id}. "
                f"ensure params are initialized before calling select."
            )
        choice = self.protocol.select(paramstate, context)
        return choice

    @callback.register()
    def train(self, context: Context, choice: int, reward: Union[int, float]):
        paramstate = self.param_backend.get(experiment_id=self.experiment_id)
        if paramstate is None:
            raise RuntimeError(
                f"param state not found for experiment {self.experiment_id}. "
                f"ensure params are initialized before calling train."
            )
        paramstate = self.protocol.train(
            ps=paramstate,
            context=context,
            choice=choice,
            reward=reward
        )
        self.param_backend.set(experiment_id=self.experiment_id, params=paramstate)
        return paramstate
