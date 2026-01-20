from typing import Any, Literal

from pydantic import BaseModel, Field
from .base import ArmConfig

OperatorType = Literal[
    "equals",
    "==",
    "eq",
    "not_equals",
    "!=",
    "ne",
    "greater_than",
    ">",
    "gt",
    "less_than",
    "<",
    "lt",
    "greater_or_equal",
    ">=",
    "gte",
    "less_or_equal",
    "<=",
    "lte",
    "contains",
    "not_contains",
    "in",
    "not_in",
]


class Rule(BaseModel):
    """Single filtering rule for metadata-based arm selection.

    Attributes
    ----------
    key : str
        Key in the metadata dictionary to evaluate
    operator : OperatorType
        Comparison operator to apply
    value : Any
        Value to compare against
    arm : ArmConfig
        Arm to commit if the rule matches
    """

    key: str = Field(..., description="Key in the data dictionary to evaluate")
    operator: OperatorType = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")
    arm: ArmConfig = Field(
        default=None, description="Arm to commit if the rule matches"
    )

    def eval(self, metadata: dict) -> bool:
        """Evaluate rule against metadata dictionary.

        Parameters
        ----------
        metadata : dict
            Dictionary containing metadata to evaluate

        Returns
        -------
        bool
            True if rule matches, False otherwise

        Notes
        -----
        Returns False if:
        - metadata is not a dict
        - key is not in metadata
        - comparison raises TypeError or AttributeError
        """
        if not isinstance(metadata, dict):
            return False

        if self.key not in metadata:
            return False

        actual = metadata[self.key]
        expected = self.value

        op = self.operator.lower()

        try:
            match op:
                case "equals" | "==" | "eq":
                    return actual == expected
                case "not_equals" | "!=" | "ne":
                    return actual != expected
                case "greater_than" | ">" | "gt":
                    return actual > expected
                case "less_than" | "<" | "lt":
                    return actual < expected
                case "greater_or_equal" | ">=" | "gte":
                    return actual >= expected
                case "less_or_equal" | "<=" | "lte":
                    return actual <= expected
                case "contains":
                    return expected in actual
                case "not_contains":
                    return expected not in actual
                case "in":
                    return actual in expected
                case "not_in":
                    return actual not in expected
                case _:
                    raise ValueError(f"Unknown operator: {self.operator}")
        except (TypeError, AttributeError):
            return False
