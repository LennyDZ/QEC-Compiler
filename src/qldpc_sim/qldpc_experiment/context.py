from abc import ABC, abstractmethod
from ast import Set
from functools import cached_property
from typing import Dict, List, Tuple
from uuid import UUID

from pydantic import BaseModel, Field


from ..data_structure import LogicalOperator, LogicalQubit
from ..qec_code import ErrorCorrectionCode
from .quantum_memory import QuantumMemory
from .record import EventType, MeasurementRecord
from .pauli_frame import FrameState


class Context(BaseModel):
    """Context model for qLDPC simulation.

    Attributes:
        logical_qubits (List[LogicalQubit]): List of logical qubits in the code.
        codes (List[ErrorCorrectionCode]): List of error correction codes used in the simulation.
        initial_assignement (Dict[UUID, ErrorCorrectionCode]): Initial assignment of logical qubits to error correction codes.
        record (MeasurementRecord): Record of measurements performed during the simulation.
        memory (QuantumMemory): Quantum memory used in the simulation.
    """

    logical_qubits: List[LogicalQubit]
    codes: List[ErrorCorrectionCode]
    initial_assignement: Dict[LogicalOperator, ErrorCorrectionCode]
    record: MeasurementRecord = Field(default_factory=MeasurementRecord)
    memory: QuantumMemory = Field(default_factory=QuantumMemory)
    frame_tracker: FrameState = Field(default_factory=FrameState)

    @cached_property
    def map_operator_to_qubits(self) -> Dict[LogicalOperator, List[LogicalQubit]]:
        """Build a mapping from logical operators to the logical qubits they act on."""
        operator_to_qubits = {}
        for q in self.logical_qubits:
            operator_to_qubits[q.logical_x] = q
            operator_to_qubits[q.logical_z] = q
        return operator_to_qubits
