from typing import List
from pydantic import BaseModel, PrivateAttr, ConfigDict, Field
from uuid import uuid4, UUID


class BaseParamState(BaseModel):

    _names: List[str] = PrivateAttr()
    _id: UUID = PrivateAttr(default_factory=uuid4)

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, validate_assignment=True
    )

    num_arms: int = Field(..., gt=0)

    def model_post_init(self, __context):
        self._names = [name for name in self.__class__.model_fields.keys()]

    @property
    def names(self) -> List[str]:
        return self._names

    @property
    def id(self) -> UUID:
        return self._id

    @classmethod
    def init(cls, **params):
        return cls.model_validate(params)
