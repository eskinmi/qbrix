from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class Arm:
    name: str
    id: str = field(default_factory=lambda: str(uuid4()))
    is_active: bool = True

    def deactivate(self) -> None:
        self.is_active = False


@dataclass
class Pool:
    name: str
    id: str = field(default_factory=lambda: str(uuid4()))
    arms: list[Arm] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.arms) == 0

    def add_arm(self, arm: Arm) -> None:
        self.arms.append(arm)

    def remove_arm(self, arm: Arm) -> None:
        self.arms.remove(arm)

    def __iter__(self):
        return iter(self.arms)

    def __len__(self) -> int:
        return len(self.arms)
