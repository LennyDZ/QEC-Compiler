from abc import abstractmethod
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pyparsing import ABC

from qldpc_sim.data_structure.tanner_graph import CheckNode
from qldpc_sim.qec_code.ec_code import ErrorCorrectionCode
from ..data_structure import LogicalOperator, TannerGraph, PauliChar, PauliEigenState
from .context import Context
from .quantum_memory import QuantumMemory
from .compilers import (
    ApplyGates,
    Compiler,
    MeasurementCompiler,
    StabilisersMeasurementCompiler,
)


class QECGadget(ABC, BaseModel):
    model_config = ConfigDict(frozen=True)
    context: Context
    tag: str = Field(default="", init=False)

    @abstractmethod
    def build_compiler_instructions(self, memory: QuantumMemory) -> List[Compiler]:
        pass


class CodeGadget(QECGadget):
    code: ErrorCorrectionCode


class LogicGadget(QECGadget):
    logical_targets: List[LogicalOperator]


class LogicalPauli(LogicGadget):
    """Logical Pauli application gadget."""

    def build_compiler_instructions(self) -> List[Compiler]:
        compilers = []
        # TODO: check compatibility of the differents logical operators (e.g. the support must not overlap (or only under specific conditions)
        for lop in self.logical_targets:
            compiler = ApplyGates(
                target_nodes=lop.target_nodes, gates=[lop.logical_type], tag=self.tag
            )
            compilers.append(compiler)
        return compilers


class StabMeasurement(CodeGadget):
    round: int = Field(
        default=1,
        description="Number of round of stabiliser measurement to perform.",
    )

    def build_compiler_instructions(self) -> List[Compiler]:

        compilers = [
            StabilisersMeasurementCompiler(
                data=self.code.tanner_graph,
                round=self.round,
                tag=f"StabMeasurement_{self.tag}",
            )
        ]
        return compilers


class InitializeCode(CodeGadget):
    """Initialize a code in a given state.
    For code with multiple logical qubits, we only support initializing in the |0> state.
    """

    initial_state: PauliEigenState = Field(default=PauliEigenState.Z_plus)

    def build_compiler_instructions(self) -> List[Compiler]:
        if len(self.code.logical_qubits) > 1:
            if self.initial_state != PauliEigenState.Z_plus:
                raise ValueError(
                    "Init only supported for |0> state for codes with multiple logical qubits."
                )
        compilers = [
            ApplyGates(
                target_nodes=self.code.tanner_graph.variable_nodes
                | self.code.tanner_graph.check_nodes,
                gates=self.initial_state.pauli_from_zero(),
                tag=f"Init_{self.tag}_reset",
            )
        ]
        stabiliser = StabilisersMeasurementCompiler(
            data=self.code.tanner_graph,
            round=1,
            tag=f"Init_{self.tag}_stab_meas",
            observable_included={
                f"z_stabs_{self.tag}": {
                    n
                    for n in self.code.tanner_graph.check_nodes
                    if n.check_type == CheckNode.CheckType.Z
                },
                f"x_stabs_{self.tag}": {
                    n
                    for n in self.code.tanner_graph.check_nodes
                    if n.check_type == CheckNode.CheckType.X
                },
            },
        )

        compilers.append(stabiliser)
        return compilers


class LM(LogicGadget):
    """Measure (project) a logical operator in the given basis. This doesn't reset nor free the qubits after measurement, so it can be used for mid-circuit measurements."""

    tag: str = Field(default="Logical Measurement", init=False)
    basis: PauliChar = Field(default=PauliChar.Z)

    def build_compiler_instructions(self) -> List[Compiler]:
        compilers = []
        lop = self.logical_targets[0]
        # Build tanner with only var nodes corresponding to the logical qubit
        t = TannerGraph(
            variable_nodes=set(lop.target_nodes),
            check_nodes=set(),
            edges=set(),
        )
        compilers.append(
            MeasurementCompiler(
                data=t,
                tag=self.tag,
                basis=self.basis,
                free_qubits=False,
                reset_qubits=False,
            )
        )
        return compilers


class Readout(CodeGadget):
    """Readout all the qubits of a code. This doesn't reset the qubits after measurement, so it can be used for mid-circuit measurements."""

    tag: str = Field(default="Readout")
    basis: PauliChar = Field(default=PauliChar.Z)

    def build_compiler_instructions(self) -> List[Compiler]:
        compilers = []
        compilers.append(
            MeasurementCompiler(
                data=self.code.tanner_graph,
                tag=self.tag,
                basis=self.basis,
                reset_qubits=False,
            )
        )
        return compilers


class PauliMeasurement(LogicGadget):
    controlled_action: LogicalPauli = Field(default=None, init=False)
