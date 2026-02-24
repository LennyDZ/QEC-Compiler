from abc import ABC, abstractmethod
from typing import Dict, List
from uuid import UUID

from pydantic import BaseModel

from ..data_structure import LogicalOperator, LogicalQubit
from ..qec_code import ErrorCorrectionCode
from .quantum_memory import QuantumMemory
from .record import MeasurementRecord


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
    record: MeasurementRecord = MeasurementRecord()
    memory: QuantumMemory = QuantumMemory()

    # def compile(
    #     self, program: List[QECGadget], optimizer: "Optimizer", compiler: "Compiler"
    # ) -> List[str]:
    #     """Compile a list of QEC gadgets into a list of Stim instructions"""
    #     compilers = []
    #     optimized_program = optimizer.optimize(program)
    #     for gadget in optimized_program:
    #         compilers.append(gadget.build_compiler_instructions())

    #     instructions = []
    #     for c in compilers:
    #         instr, event_tags = c.compile(self.memory)
    #         instructions.extend(instr)
    #         if event_tags:
    #             for tag in event_tags:
    #                 self.record.add_measurement_record(tag)
    #     return instructions

    # def build_detector_error_model(self) -> List[str]:
    #     """Build a detector error model from the record of measurements."""
    #     for record in self.record.measurement_records:
    #         record.build_detector_error_model()
    #     return []


# class Compiler(ABC):
#     """Abstract base class for compilers in qLDPC simulation."""

#     # TODO: Somehow provide a way to plug in specific compilers for a defined set of lower level operations (stabiliaser measurement, logical pauli application, etc.) and specific codes (e.g. hypergraph product code, etc.)
#     @abstractmethod
#     def compile(self, memory: QuantumMemory) -> List[str]:
#         """Compile the operation into a list of stim instructions."""
#         pass


# class Optimizer(ABC, BaseModel):
#     """Optimizer for qLDPC simulation.

#     Attributes:
#         name (str): Name of the optimizer.
#     """

#     name: str

#     @abstractmethod
#     def optimize(self, gadget: List[QECGadget]) -> List[QECGadget]:
#         """Optimize a list of QEC gadgets"""
#         pass
