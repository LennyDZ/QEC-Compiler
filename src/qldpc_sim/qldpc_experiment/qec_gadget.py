from abc import abstractmethod
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field
from pyparsing import ABC

from qldpc_sim.qec_code.ec_code import ErrorCorrectionCode
from .pauli_frame import FrameUpdate
from .record import EventType, OutcomeSet
from ..data_structure import LogicalOperator, TannerGraph, PauliChar, PauliEigenState
from .context import Context
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
    def build_compiler_instructions(
        self,
    ) -> Tuple[List[Compiler], List[OutcomeSet]]:
        pass


class CodeGadget(QECGadget):
    code: ErrorCorrectionCode


class LogicGadget(QECGadget):
    logical_targets: List[LogicalOperator]
    frame_update: Optional[FrameUpdate] = Field(
        default=None,
        description="update to apply to the logical pauli frame when going through this gadget.",
    )


class LogicalPauli(LogicGadget):
    """Logical Pauli application gadget."""

    def build_compiler_instructions(
        self,
    ) -> Tuple[List[Compiler], List[OutcomeSet]]:
        compilers = []
        # TODO: check compatibility of the differents logical operators (e.g. the support must not overlap (or only under specific conditions)
        for lop in self.logical_targets:
            compiler = ApplyGates(
                target_nodes=lop.target_nodes, gates=[lop.logical_type], tag=self.tag
            )
            compilers.append(compiler)
        return compilers, []


class StabMeasurement(CodeGadget):
    round: int = Field(
        default=1,
        description="Number of round of stabiliser measurement to perform.",
    )

    def build_compiler_instructions(
        self,
    ) -> Tuple[List[Compiler], List[OutcomeSet]]:

        compilers = [
            StabilisersMeasurementCompiler(
                data=self.code.tanner_graph,
                round=self.round,
                tag=f"StabMeasurement_{self.tag}",
            )
        ]
        outcomes = [
            OutcomeSet(
                tag=f"StabMeasurement_{self.tag}",
                type=EventType.STAB_MEASUREMENT,
                size=len(self.code.tanner_graph.check_nodes),
                measured_nodes=set(self.code.tanner_graph.check_nodes),
            )
        ]
        return compilers, outcomes


class InitializeCode(CodeGadget):
    """Initialize a code in a given state.
    For code with multiple logical qubits, we only support initializing in the |0> state.
    """

    initial_state: PauliEigenState = Field(default=PauliEigenState.Z_plus)

    def build_compiler_instructions(
        self,
    ) -> Tuple[List[Compiler], List[OutcomeSet]]:
        if len(self.code.logical_qubits) > 1:
            if self.initial_state != PauliEigenState.Z_plus:
                raise ValueError(
                    "Init only supported for |0> state for codes with multiple logical qubits."
                )
        compilers = []
        if (
            self.initial_state == PauliEigenState.X_plus
            or self.initial_state == PauliEigenState.X_minus
        ):
            compilers.append(
                ApplyGates(
                    target_nodes=self.code.tanner_graph.variable_nodes,
                    gates=["H"],
                    tag=f"{self.tag}_logical_pauli",
                )
            )

        stabiliser = StabilisersMeasurementCompiler(
            data=self.code.tanner_graph,
            round=1,
            tag=f"{self.tag}_stab_meas",
            observable_included={
                f"z_stabs_{self.tag}": {
                    n
                    for n in self.code.tanner_graph.check_nodes
                    if n.check_type == PauliChar.Z
                },
                f"x_stabs_{self.tag}": {
                    n
                    for n in self.code.tanner_graph.check_nodes
                    if n.check_type == PauliChar.X
                },
            },
        )
        compilers.append(stabiliser)

        match self.initial_state:
            case PauliEigenState.Z_minus:
                compilers.append(
                    ApplyGates(
                        target_nodes=self.code.logical_qubits[0].logical_x.target_nodes,
                        gates=[PauliChar.X],
                        tag=f"LX_{self.tag}_logical_pauli",
                    )
                )
            case PauliEigenState.X_minus:
                compilers.append(
                    ApplyGates(
                        target_nodes=self.code.logical_qubits[0].logical_z.target_nodes,
                        gates=[PauliChar.Z],
                        tag=f"LZ_{self.tag}_logical_pauli",
                    )
                )
        return compilers, []


class LM(LogicGadget):
    """Measure (project) a logical operator in the given basis. This doesn't reset nor free the qubits after measurement, so it can be used for mid-circuit measurements."""

    tag: str = Field(default="Logical Measurement", init=False)
    basis: PauliChar = Field(default=PauliChar.Z)

    def build_compiler_instructions(self) -> Tuple[List[Compiler], List[OutcomeSet]]:
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
        outcomes = [
            OutcomeSet(
                tag=f"LogicalMeasurement_{self.tag}",
                type=EventType.OBSERVABLE,
                size=len(lop.target_nodes),
                measured_nodes=set(lop.target_nodes),
                target=lop,
            )
        ]
        return compilers, outcomes


class Readout(CodeGadget):
    """Readout all the qubits of a code. This doesn't reset the qubits after measurement, so it can be used for mid-circuit measurements."""

    tag: str = Field(default="Readout")
    basis: PauliChar = Field(default=PauliChar.Z)

    def build_compiler_instructions(self) -> Tuple[List[Compiler], List[OutcomeSet]]:
        compilers = []
        compilers.append(
            MeasurementCompiler(
                data=self.code.tanner_graph,
                tag=self.tag,
                basis=self.basis,
                reset_qubits=False,
            )
        )
        outcomes = [
            OutcomeSet(
                tag=f"Readout_{self.tag}",
                type=EventType.OBSERVABLE,
                size=len(self.code.tanner_graph.variable_nodes),
                measured_nodes=set(self.code.tanner_graph.variable_nodes),
            )
        ]
        return compilers, outcomes


class PauliMeasurement(LogicGadget):
    pass
