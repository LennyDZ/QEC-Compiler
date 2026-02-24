from __future__ import annotations
from typing import List, Set, Tuple
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .pauli import PauliChar, PauliString
from .tanner_graph import VariableNode


class LogicalOperator(BaseModel):
    """Class representing a logical operator in a quantum error correction code.

    Attributes:
        logical_type: The type of logical operator (X, Y, or Z).
        operator: The Pauli string representing the physical implementation of the logical operator (each element of the string applies to a corresponding variable node in the target_nodes tuple).
        target_nodes: The tuple of variable nodes the operator acts on.

    """

    model_config = ConfigDict(frozen=True)  # Immutable after creation
    id: UUID = Field(default_factory=uuid4, init=False)
    logical_type: PauliChar
    operator: PauliString
    target_nodes: Tuple[
        VariableNode, ...
    ]  # Tuple of variable nodes the operator acts on.

    @model_validator(mode="after")
    def ensure_length_match(self) -> "LogicalOperator":
        """Ensure that the length of the Pauli string matches the number of target variable nodes."""
        if len(self.operator.string) != len(self.target_nodes):
            raise ValueError(
                "Length of Pauli string must match number of target variable nodes."
            )
        return self
