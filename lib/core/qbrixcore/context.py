import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Context:
    """Request context for bandit selection.

    The vector field accepts both list[float] and np.ndarray. Contextual policies
    (LinUCB, LinTS) will convert lists to arrays internally when needed.
    Stochastic policys ignore the vector entirely.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vector: np.ndarray | list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
