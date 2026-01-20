from __future__ import annotations

import random
import uuid
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import ClassVar

import gevent
from locust import User
from locust import between
from locust import events
from locust import task

from tests.loadtest.client import ProxyClient
from tests.loadtest.client import SelectResult
from tests.loadtest.config import settings


@dataclass
class PendingFeedback:
    request_id: str


@dataclass
class ExperimentInfo:
    experiment_id: str
    pool_id: str
    user_count: int = 0


class MultiExperimentUser(User):
    """
    simulates users distributed across multiple experiments.

    behavior:
    - multiple experiments are created at test start
    - each user is assigned to one experiment (round-robin with max cap)
    - users interact only with their assigned experiment
    - same realistic async feedback behavior as single experiment

    this simulates real-world multi-tenant scenarios where:
    - different user segments have different experiments
    - experiments have user capacity limits
    - system handles concurrent load across experiments
    """

    wait_time = between(0.1, 0.5)

    # shared state across all users
    experiments: ClassVar[list[ExperimentInfo]] = []
    _setup_done: ClassVar[bool] = False
    _assignment_lock: ClassVar[Lock] = Lock()
    _user_counter: ClassVar[int] = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client: ProxyClient | None = None
        self.assigned_experiment: ExperimentInfo | None = None
        self.pending_feedbacks: deque[PendingFeedback] = deque(maxlen=100)

    def on_start(self) -> None:
        self.client = ProxyClient()
        self.client.connect()
        self._assign_experiment()

    def on_stop(self) -> None:
        if self.client:
            self.client.close()
        self._release_experiment()

    def _assign_experiment(self) -> None:
        """assign this user to an experiment with capacity."""
        with MultiExperimentUser._assignment_lock:
            # find experiment with available capacity
            for exp in MultiExperimentUser.experiments:
                if exp.user_count < settings.max_users_per_experiment:
                    exp.user_count += 1
                    self.assigned_experiment = exp
                    return

            # if all at capacity, assign to least loaded
            if MultiExperimentUser.experiments:
                exp = min(MultiExperimentUser.experiments, key=lambda e: e.user_count)
                exp.user_count += 1
                self.assigned_experiment = exp

    def _release_experiment(self) -> None:
        """release user slot from assigned experiment."""
        if self.assigned_experiment:
            with MultiExperimentUser._assignment_lock:
                self.assigned_experiment.user_count = max(
                    0, self.assigned_experiment.user_count - 1
                )

    @task(10)
    def select_arm(self) -> None:
        """make a selection request on assigned experiment."""
        if not self.assigned_experiment or not self.client:
            return

        context_id = str(uuid.uuid4())
        context_vector = [random.random() for _ in range(settings.context_vector_dim)]
        context_metadata = {
            "device": random.choice(["mobile", "desktop", "tablet"]),
            "region": random.choice(["us-east", "us-west", "eu", "apac"]),
            "user_tier": random.choice(["free", "premium", "enterprise"]),
        }

        try:
            result: SelectResult = self.client.select(
                experiment_id=self.assigned_experiment.experiment_id,
                context_id=context_id,
                context_vector=context_vector,
                context_metadata=context_metadata,
            )

            events.request.fire(
                request_type="grpc",
                name="Select",
                response_time=0,
                response_length=0,
                exception=None,
                context={},
            )

            # probabilistically queue feedback
            if random.random() < settings.feedback_probability:
                pending = PendingFeedback(request_id=result.request_id)
                self.pending_feedbacks.append(pending)

                delay_ms = random.randint(
                    settings.feedback_delay_min_ms,
                    settings.feedback_delay_max_ms,
                )
                gevent.spawn_later(delay_ms / 1000.0, self._send_feedback, pending)

        except Exception as e:
            events.request.fire(
                request_type="grpc",
                name="Select",
                response_time=0,
                response_length=0,
                exception=e,
                context={},
            )

    @task(1)
    def health_check(self) -> None:
        """periodic health check."""
        if not self.client:
            return

        try:
            self.client.health_check()
            events.request.fire(
                request_type="grpc",
                name="Health",
                response_time=0,
                response_length=0,
                exception=None,
                context={},
            )
        except Exception as e:
            events.request.fire(
                request_type="grpc",
                name="Health",
                response_time=0,
                response_length=0,
                exception=e,
                context={},
            )

    def _send_feedback(self, pending: PendingFeedback) -> None:
        """send feedback asynchronously."""
        if not self.client or not self.client.is_connected:
            return

        reward = 1.0 if random.random() < settings.reward_success_probability else 0.0

        try:
            self.client.feedback(
                request_id=pending.request_id,
                reward=reward,
            )
            events.request.fire(
                request_type="grpc",
                name="Feedback",
                response_time=0,
                response_length=0,
                exception=None,
                context={},
            )
        except Exception as e:
            events.request.fire(
                request_type="grpc",
                name="Feedback",
                response_time=0,
                response_length=0,
                exception=e,
                context={},
            )


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """setup multiple experiments before test starts."""
    if MultiExperimentUser._setup_done:
        return

    client = ProxyClient()
    client.connect()

    try:
        for i in range(settings.num_experiments):
            # create pool
            pool_id = client.create_pool(
                name=f"{settings.default_pool_name}-multi-{i}-{uuid.uuid4().hex[:8]}",
                num_arms=settings.default_num_arms,
            )

            # create experiment
            experiment_id = client.create_experiment(
                name=f"{settings.default_experiment_name}-multi-{i}-{uuid.uuid4().hex[:8]}",
                pool_id=pool_id,
                protocol=settings.default_protocol,
            )

            MultiExperimentUser.experiments.append(
                ExperimentInfo(experiment_id=experiment_id, pool_id=pool_id)
            )

            print(
                f"created experiment {i + 1}/{settings.num_experiments}: {experiment_id}"
            )

        MultiExperimentUser._setup_done = True

    finally:
        client.close()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """cleanup all experiments after test ends."""
    if not MultiExperimentUser._setup_done:
        return

    client = ProxyClient()
    client.connect()

    try:
        for exp in MultiExperimentUser.experiments:
            try:
                client.delete_experiment(exp.experiment_id)
                print(f"deleted experiment: {exp.experiment_id}")
            except Exception as e:
                print(f"failed to delete experiment {exp.experiment_id}: {e}")

            try:
                client.delete_pool(exp.pool_id)
                print(f"deleted pool: {exp.pool_id}")
            except Exception as e:
                print(f"failed to delete pool {exp.pool_id}: {e}")

    finally:
        client.close()
        MultiExperimentUser._setup_done = False
        MultiExperimentUser.experiments.clear()
