from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ..data_structure import (
    CheckNode,
    TannerNode,
    PauliEigenState,
    PauliChar,
    TannerGraph,
)
from .record import EventTag, EventType
from .quantum_memory import QuantumMemory


class Compiler(ABC, BaseModel):
    """Abstract base class for compilable entities in qLDPC simulation.
    A Compiler takes as input a Tanner graph and provides methods to evaluate costs and compile a given operation.
    """

    data: TannerGraph
    tag: str = ""

    @abstractmethod
    def qubits_cost(self) -> int:
        """Returns the number of qubits required to perform the operation."""
        return self.data.number_of_nodes

    def check_enough_memory(self, memory: QuantumMemory) -> None:
        """Checks if there are enough qubits in the quantum memory to compile the operation."""
        nodes_uid = {
            n.id for n in self.data.variable_nodes.union(self.data.check_nodes)
        }
        allocated_uid = set(memory.allocation.keys())
        if len(nodes_uid - allocated_uid) > memory.number_of_available_qubits:
            raise ValueError(
                "Not enough qubits are available in the quantum memory to compile the stabiliser measurements."
                f" Needed: {len(nodes_uid - allocated_uid)}, Available: {memory.number_of_available_qubits}"
            )
        return None

    @abstractmethod
    def gate_cost(self) -> Tuple[int, int]:
        """Returns a tuple (number_of_single_qubit_gates, number_of_two_qubit_gates)."""
        pass

    @abstractmethod
    def compile(
        self, memory: QuantumMemory
    ) -> Tuple[List[str], Optional[List[EventTag]]]:
        """Compiles the operation into a list of stim instructions and an optional list of event tags."""
        pass


class ApplyGates(Compiler):
    """Compiler for applying a sequence of gates to a set of targeted nodes."""

    data: TannerGraph = TannerGraph(
        variable_nodes=set(), check_nodes=set(), edges=set()
    )
    target_nodes: List[TannerNode]
    gates: List[str]  # e.g. ["H"], ["X", "H"], etc.

    def compile(
        self, memory: QuantumMemory
    ) -> Tuple[List[str], Optional[List[EventTag]]]:
        self.check_enough_memory(memory)
        stim_instructions = []
        for dq in self.target_nodes:
            if dq.id not in memory.allocation:
                memory.allocate_qubit(dq.id)
            for gate in self.gates:
                stim_instructions.append(f"{gate} {memory.allocation[dq.id]}")
        return stim_instructions, None

    def gate_cost(self):
        return self.data.number_of_nodes * len(self.gates)

    def qubits_cost(self):
        return len(self.target_nodes)


class MeasurementCompiler(Compiler):
    """Class representing the compiler for measurement of all qubits in a Tanner graph."""

    basis: PauliChar = Field(
        default=PauliChar.Z,
        description="Measurement basis for the destructive measurement. Can be X, Y, or Z.",
    )
    free_qubits: bool = Field(
        default=True, description="Whether to free the qubits after measurement."
    )
    reset_qubits: bool = Field(
        default=True,
        description="Whether to reset the qubits after measurement. If True, the qubits will be reset to |0> state after measurement.",
    )

    def gate_cost(self):
        """gate cost for measuring all qubits in the Tanner graph."""
        return self.data.number_of_nodes()

    def qubits_cost(self):
        return 0

    def compile(self, memory: QuantumMemory) -> Tuple[List[str], EventTag]:
        self.check_enough_memory(memory)
        stim_instructions = []
        node_measured = []
        # If free qubits is true, we have to reset.
        if self.reset_qubits or self.free_qubits:
            r_tag = "R"
        else:
            r_tag = ""
        for q in self.data.variable_nodes | self.data.check_nodes:
            if q.id not in memory.allocation:
                raise ValueError(
                    f"Qubit {q} is not allocated in the quantum memory. Cannot compile measurement instruction."
                )
            else:
                match self.basis:
                    case PauliChar.X:
                        stim_instructions.append(f"M{r_tag}X {memory.allocation[q.id]}")
                    case PauliChar.Y:
                        stim_instructions.append(f"M{r_tag}Y {memory.allocation[q.id]}")
                    case PauliChar.Z:
                        stim_instructions.append(f"M{r_tag}Z {memory.allocation[q.id]}")
                    case _:
                        raise ValueError(f"Unknown measurement basis: {self.basis}")
                node_measured.append(q)

                if self.free_qubits:
                    memory.free_qubit(q.id)
        return stim_instructions, EventTag(
            tag=f"{self.tag}",
            type=EventType.OBSERVABLE,
            size=self.data.number_of_nodes,
            measured_nodes=node_measured,
        )


class StabilisersMeasurementCompiler(Compiler):
    """Class representing the compiler for measurement of stabilisers in a quantum error-correcting code."""

    round: int
    observable_included: Dict[str, Set[CheckNode]] = Field(
        default_factory=dict,
        description="Observables included in the stabiliser measurement. Each of them is represented as a set of the Check nodes that are included in the observable.",
    )
    check_initial_state: PauliEigenState = Field(
        default=PauliEigenState.Z_plus,
        description="Initial state for the ancilla qubits used in stabiliser checks.",
    )

    def gate_cost(self):
        """gate cost for measuring all stabilisers of the tanner <round> times. For the single qubits, it is a upper bound"""
        single_qubit_gates = 0
        two_qubit_gates = 0

        for c in self.data.check_nodes:
            deg = self.data.degree(c)
            single_qubit_gates += (
                2 + deg
            ) * self.round  # H gates for X stabilisers + MZ
            two_qubit_gates += deg * self.round  # CX gates

    def qubits_cost(self):
        return 0

    def compile(self, memory: QuantumMemory) -> Tuple[List[str], List[EventTag]]:
        check_nodes = self.data.check_nodes
        base_stim_instructions = []
        measured = []  # keep track of the order of measurements for the event tag

        for check in check_nodes:
            if check.id not in memory.allocation:
                memory.allocate_qubit(check.id)
            base_stim_instructions.append(f"RZ {memory.allocation[check.id]}")
            for dq in self.data.get_neighbourhood(check):
                if dq.id not in memory.allocation:
                    memory.allocate_qubit(dq.id)

        for check in check_nodes:
            base_stim_instructions.append(
                f"# Stab: {check.tag}, type: {check.check_type}"
            )  # debug
            match (check.check_type):
                case "Z":
                    for dq in self.data.get_neighbourhood(check):
                        base_stim_instructions.append(
                            f"CX {memory.allocation[dq.id]} {memory.allocation[check.id]}"
                        )
                case "X":
                    base_stim_instructions.append(f"H {memory.allocation[check.id]}")
                    for dq in self.data.get_neighbourhood(check):
                        base_stim_instructions.append(
                            f"CX {memory.allocation[check.id]} {memory.allocation[dq.id]}"
                        )
                    base_stim_instructions.append(f"H {memory.allocation[check.id]}")
                case "Y":
                    base_stim_instructions.append(f"H {memory.allocation[check.id]}")
                    for dq in self.data.get_neighbourhood(check):
                        base_stim_instructions.append(
                            f"CY {memory.allocation[check.id]}, {memory.allocation[dq.id]}"
                        )
                    base_stim_instructions.append(f"H {memory.allocation[check.id]}")
                case "Mixed":
                    edge = self.data.index_by_check(check)
                    for e in edge:
                        dq = e.variable_node
                        if e.pauli_check == "Z":
                            base_stim_instructions.append(
                                f"CX {memory.allocation[dq.id]} {memory.allocation[check.id]}"
                            )
                        elif e.pauli_check == "X":
                            base_stim_instructions.append(
                                f"H {memory.allocation[check.id]}"
                            )
                            base_stim_instructions.append(
                                f"CX {memory.allocation[check.id]} {memory.allocation[dq.id]}"
                            )
                            base_stim_instructions.append(
                                f"H {memory.allocation[check.id]}"
                            )
                        elif e.pauli_check == "Y":
                            base_stim_instructions.append(
                                f"H {memory.allocation[check.id]}"
                            )
                            base_stim_instructions.append(
                                f"CY {memory.allocation[check.id]}, {memory.allocation[dq.id]}"
                            )
                            base_stim_instructions.append(
                                f"H {memory.allocation[check.id]}"
                            )
                        else:
                            raise ValueError(
                                f"Unknown pauli check type: {e.pauli_check}"
                            )

                case _:
                    raise ValueError(f"Unknown stabiliser type: {check.check_type}")

        for check in check_nodes:
            base_stim_instructions.append(f"MZ {memory.allocation[check.id]}")
            measured.append(check)
        tab = " " * 4
        stim_instructions = (
            [f"REPEAT {self.round} {{"]
            + [tab + i for i in base_stim_instructions]
            + ["}"]
        )

        events = [
            EventTag(
                tag=f"{self.tag}_round{i}",
                type=EventType.STAB_MEASUREMENT,
                size=len(check_nodes),
                measured_nodes=measured.copy(),
            )
            for i in range(self.round)
        ]
        if self.observable_included:
            for i, (key, obs) in enumerate(self.observable_included.items()):
                events.append(
                    EventTag(
                        tag=f"{self.tag}_observable_{key}",
                        type=EventType.OBSERVABLE,
                        size=0,
                        measured_nodes=list(obs),
                    )
                )

        return (stim_instructions, events)
