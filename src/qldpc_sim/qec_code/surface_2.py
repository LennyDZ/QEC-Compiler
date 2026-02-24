from typing import List
import numpy as np
from pydantic import Field, model_validator
from scipy.sparse import csr_matrix

from qldpc_sim.data_structure.logical_operator import LogicalOperator
from qldpc_sim.data_structure.logical_qubit import LogicalQubit
from qldpc_sim.data_structure.pauli import PauliChar, PauliString
from qldpc_sim.data_structure.tanner_graph import TannerGraph
from .ec_code import ErrorCorrectionCode


class RSC2(ErrorCorrectionCode):
    """Rotated Surface Code with distance 3.

    The rotated surface code uses a 3x3 grid of qubits with:
    - 9 data qubits
    - 8 plaquette operators (X and Z stabilizers)
    - 1 logical qubit
    - Distance 3
    """

    name: str = "RSC3"
    n: int = Field(
        default=5, init=False, description="Number of physical qubits in the code."
    )
    k: int = Field(
        default=1, init=False, description="Number of logical qubits in the code."
    )
    d: int = Field(default=2, init=False, description="Distance of the code.")
    validate_algebraic_properties: bool = False
    tanner_graph: TannerGraph | None = None
    logical_qubits: List[LogicalQubit] = []

    @model_validator(mode="after")
    def compute_tanner_graph(self) -> "RSC2":
        """Construct the Tanner graph for the rotated surface code distance 3."""
        # Parity check matrices for rotated surface code (3x3 grid)
        # X stabilizers (plaquette operators)
        Hx = np.array(
            [[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]],
            dtype=int,
        )

        # Z stabilizers (vertex operators)
        Hz = np.array(
            [[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]],
            dtype=int,
        )
        self.tanner_graph = TannerGraph.from_pcm(
            csr_matrix(Hx), csr_matrix(Hz), code_name=self.name
        )
        return self

    @model_validator(mode="after")
    def compute_logical_qubits(self) -> "RSC2":
        """Define logical operators as simple boundary operators."""
        var_nodes = tuple(sorted(self.tanner_graph.variable_nodes, key=lambda v: v.tag))
        # Logical Z: all qubits in a row (simple boundary)
        logical_z_nodes = var_nodes[:2]
        logical_z = LogicalOperator(
            operator=PauliString(string=(PauliChar.Z,) * len(logical_z_nodes)),
            target_nodes=logical_z_nodes,
            logical_type=PauliChar.Z,
        )

        # Logical X: all qubits in a column (simple boundary)
        logical_x_nodes = [var_nodes[0], var_nodes[3]]
        logical_x = LogicalOperator(
            operator=PauliString(string=(PauliChar.X,) * len(logical_x_nodes)),
            target_nodes=logical_x_nodes,
            logical_type=PauliChar.X,
        )

        self.logical_qubits = [
            LogicalQubit(
                name="lq0",
                logical_x=logical_x,
                logical_z=logical_z,
            )
        ]
        return self
