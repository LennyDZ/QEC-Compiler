from collections import defaultdict
from functools import cached_property
from typing import Dict, List, Set, Tuple
from pydantic import BaseModel, ConfigDict

from ..data_structure import (
    TannerGraph,
    LogicalOperator,
    CheckNode,
    TannerEdge,
    TannerNode,
    VariableNode,
    PauliEigenState,
    PauliChar,
)
from ..data_structure import TannerGraphAlgebra as tga
from ..qldpc_experiment import (
    PauliMeasurement,
    ApplyGates,
    MeasurementCompiler,
    StabilisersMeasurementCompiler,
)


class SurgeJoint(BaseModel):
    model_config = ConfigDict(frozen=True)
    check_type: PauliChar
    stack: List[TannerNode]


class SurgeMeasurement(PauliMeasurement):

    distance: int

    @cached_property
    def tanner_supports(self) -> Dict[LogicalOperator, TannerGraph]:
        tanners = {}
        for lop in self.logical_targets:
            code = self.context.initial_assignement[lop]
            new_support = code.tanner_graph.get_support(
                lop.target_nodes, lop.logical_type
            )
            new_key = lop
            for key, other_support in tanners.items():
                # Check if supports are disjoint or not
                # All joint support are merged together, as they will be measured by the same ancilla structure.

                if not new_support.is_disjoint(other_support):
                    raise ValueError(
                        "Error in building measurement: two logical operators have overlapping support"
                    )
                # if new_support == other_support:
                #     raise ValueError(
                #         "Error in building measurement: two logical operators have the same support",
                #         "This means either the same logical operator appear twice in the operator list, or a logical operator is fully supported by a higher weight logical operator. In both cases, the measurement is not well defined (for now at least).",
                #     )
                # else:
                #     other_support = tanners[key]
                #     new_key = key + lop if isinstance(key, tuple) else (key, lop)
                #     new_support = new_support | other_support

            tanners[new_key] = new_support
        return tanners

    def check_feasibility(self):
        # Implement feasibility checks specific to Measurement
        # Check if operators are disjoint or not, etc.
        return True

    def cost(self):
        # Implement cost evaluation specific to Measurement
        return 0

    def build_compiler_instructions(self):
        compilers = []
        # I. Evalute feasability, cost
        if not self.check_feasibility():
            raise ValueError(
                "Measurement instruction is not feasible with the given memory."
            )
        # II. Build ancilla TannerGraphs
        lop_tanners = self.tanner_supports
        tanner_codes = TannerGraph()  # Sum of tanner of codes involved (disconnected)

        joints = []
        for k, v in lop_tanners.items():
            vn = sorted(v.variable_nodes, key=lambda n: n.tag)
            bot = vn[0]

            if len(v.index_by_variable[bot]) != 1:
                bot = vn[1]
            stack = [bot]
            p = bot
            pc = None
            for i in range(1, self.distance - 1):
                nc = [
                    check for check in v.index_by_variable[p] if check.check_node != pc
                ][0].check_node
                stack.append(nc)
                nv = [var for var in v.index_by_check[nc] if var.variable_node != p]
                nv = nv[0].variable_node
                stack.append(nv)

                p = nv
                pc = nc
            joint = SurgeJoint(
                check_type=pc.check_type,
                stack=stack,
            )
            joints.append(joint)

        connecting_edges = []
        check_anticommuting_with_lop = defaultdict(set)

        c1 = self.context.initial_assignement[self.logical_targets[0]].tanner_graph
        c2 = self.context.initial_assignement[self.logical_targets[1]].tanner_graph

        bridge_tanner, bridge_edges = self._build_bridge_tanner((joints[0], joints[1]))
        tanner_codes = c1 | c2
        merged_tanner = tga.connect(
            bridge_tanner, tanner_codes, connecting_edges=bridge_edges
        )
        # V. Build compiler
        basis = self.logical_targets[0].logical_type

        if basis == PauliChar.X:
            check_type = CheckNode.CheckType.X
            var_node_initial_state = PauliEigenState.Z_plus
        elif basis == PauliChar.Z:
            check_type = CheckNode.CheckType.Z
            var_node_initial_state = PauliEigenState.X_plus
        else:
            raise ValueError("Only X and Z measurement are supported for now.")

        init_ancilla = [
            ApplyGates(
                tag=f"init_{self.tag}",
                target_nodes=bridge_tanner.variable_nodes,
                gates=var_node_initial_state.pauli_from_zero(),
            ),
        ]

        observable_nodes = {
            "XX_outcome": set(
                [n for n in bridge_tanner.check_nodes if n.check_type == check_type]
            ),
            "middle_X": set(
                [
                    n
                    for n in bridge_tanner.check_nodes
                    if len(bridge_tanner.index_by_check[n]) == 2
                ]
            ),
            "all_ancilla_checks": bridge_tanner.check_nodes,
            "all_code_checks": merged_tanner.check_nodes,
        }

        i = 0
        for k, v in check_anticommuting_with_lop.items():
            i += 1
            observable_nodes[f"anticommute_with_{i}"] = v

        stab_measurement = StabilisersMeasurementCompiler(
            data=merged_tanner,
            round=self.distance,
            tag=f"merged_stab_{self.tag}",
            observable_included=observable_nodes,
        )

        readout_ancilla = MeasurementCompiler(
            data=bridge_tanner,
            tag=f"readout_bridge_ancilla_{self.tag}",
            reset_qubits=True,
            free_qubits=True,
        )
        compilers.extend(init_ancilla)
        compilers.append(stab_measurement)

        compilers.append(readout_ancilla)
        return compilers

    def _build_bridge_tanner(
        self, bridge_ends: tuple[SurgeJoint, SurgeJoint]
    ) -> Tuple[TannerEdge, Set[TannerEdge]]:
        connecting_edges = set()
        bridge_edges = set()
        var_nodes = set()
        check_nodes = set()
        # joint are of the same type (easy case)
        if bridge_ends[0].check_type == bridge_ends[1].check_type:
            c_type = bridge_ends[0].check_type
            prev_layer_node = None
            # iter nodes connected to the bridge (at each layer)
            for i, (s1, s2) in enumerate(
                zip(bridge_ends[0].stack, bridge_ends[1].stack)
            ):
                if isinstance(s1, VariableNode) and isinstance(s2, VariableNode):
                    check_type = c_type.dual()
                    new_check = CheckNode(
                        tag=f"bridge_{check_type}_check_l{i}",
                        check_type=check_type,
                    )
                    edge1 = TannerEdge(
                        variable_node=s1,
                        check_node=new_check,
                        pauli_checked=check_type,
                    )
                    edge2 = TannerEdge(
                        variable_node=s2,
                        check_node=new_check,
                        pauli_checked=check_type,
                    )
                    if i > 0:
                        edge_with_prev_layer = TannerEdge(
                            variable_node=prev_layer_node,
                            check_node=new_check,
                            pauli_checked=check_type,
                        )
                        bridge_edges.add(edge_with_prev_layer)
                    prev_layer_node = new_check
                    connecting_edges.add(edge1)
                    connecting_edges.add(edge2)
                    check_nodes.add(new_check)

                elif isinstance(s1, CheckNode) and isinstance(s2, CheckNode):
                    c_type = s1.check_type
                    new_variable = VariableNode(
                        tag=f"bridge_var_l{i}",
                    )
                    edge1 = TannerEdge(
                        variable_node=new_variable,
                        check_node=s1,
                        pauli_checked=c_type,
                    )
                    edge2 = TannerEdge(
                        variable_node=new_variable,
                        check_node=s2,
                        pauli_checked=c_type,
                    )
                    if i > 0:
                        edge_with_prev_layer = TannerEdge(
                            variable_node=new_variable,
                            check_node=prev_layer_node,
                            pauli_checked=c_type.dual(),
                        )
                        bridge_edges.add(edge_with_prev_layer)
                    prev_layer_node = new_variable
                    connecting_edges.add(edge1)
                    connecting_edges.add(edge2)
                    var_nodes.add(new_variable)
                else:
                    raise ValueError(
                        "Error in building bridge Tanner: joint stacks are inconsistent."
                    )
        else:
            raise ValueError("Only bridge between same check types are supported yet.")

        return (
            TannerGraph(
                variable_nodes=var_nodes, check_nodes=check_nodes, edges=bridge_edges
            ),
            connecting_edges,
        )
