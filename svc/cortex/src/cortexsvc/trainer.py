from collections import defaultdict
import numpy as np

from qbrixcore.context import Context
from qbrixcore.protoc.base import BaseProtocol
from qbrixstore.redis.client import RedisClient
from qbrixstore.redis.streams import FeedbackEvent


def _build_protocol_map() -> dict[str, type[BaseProtocol]]:
    registry = {}
    def collect(cls):
        for subclass in cls.__subclasses__():
            if hasattr(subclass, "name") and subclass.name:
                registry[subclass.name] = subclass
            collect(subclass)
    collect(BaseProtocol)
    return registry


PROTOCOL_MAP = _build_protocol_map()


class BatchTrainer:

    def __init__(self, redis_client: RedisClient):
        self._redis = redis_client

    @staticmethod
    def _group_per_experiment(events: list[FeedbackEvent]):
        grouped = defaultdict(list)
        for event in events:
            grouped[event.experiment_id].append(event)
        return grouped

    async def _train_experiment(
        self, experiment_id: str, events: list[FeedbackEvent]
    ) -> int:
        experiment_record = await self._redis.get_experiment(experiment_id)
        if experiment_record is None:
            return 0

        protocol_name = experiment_record["protocol"]
        protocol_cls = PROTOCOL_MAP.get(protocol_name)
        if protocol_cls is None:
            return 0

        # attention: in a distributed setup, where cortex node has multiple replicas, this below
        #  read and train loop would be a problem (per experiment) as there would be race conditions:
        #  <->
        #  consider:
        #   ==========
        #   for experiment Ei;
        #   t1: redis read, in instance X
        #   t2: experiment train
        #   t3: redis read, in instance Y
        #   t4: redis write, trained parameters, in instance X
        #   t5: experiment train
        #   t6: redis write, trained parameters, in instance Y
        #   ==========
        #  currently the setup imposes that the cortex instance shall only have single replica as it
        #  operates on event sourcing. however, in the future, if we would like to scale the
        #  trainers (cortex instances) as well, we'd have to change the parameter update logic, as it
        #  is presented here.
        #  <->
        #  a solution to this is to make experiment level separation where; different cortex instances
        #  train different set of experiment, which likely requires additional metadata services like etcd
        #  and a coordination setup.


        params = await self._redis.get_params(experiment_id)
        if params is None:
            num_arms = len(experiment_record["pool"]["arms"])
            param_state = protocol_cls.init_params(
                num_arms=num_arms, **experiment_record.get("protocol_params", {})
            )
        else:
            param_state = protocol_cls.param_state_cls.model_validate(
                params
            )

        for event in events:
            context = Context(
                id=event.context_id,
                vector=np.array(event.context_vector, dtype=np.float16),
                metadata=event.context_metadata,
            )
            # train and retrieve the updated parameter state.
            param_state = protocol_cls.train(
                ps=param_state,
                context=context,
                choice=event.arm_index,
                reward=event.reward,
            )

        await self._redis.set_params(experiment_id, param_state.model_dump())
        return len(events)

    async def train(self, events: list[FeedbackEvent]) -> dict[str, int]:
        ledger = dict()
        grouped_events = self._group_per_experiment(events)
        for experiment_id, experiment_events in grouped_events.items():
            count = await self._train_experiment(
                experiment_id,
                experiment_events
            )
            ledger[experiment_id] = count
        return ledger
