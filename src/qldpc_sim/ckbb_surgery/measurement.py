from collections import defaultdict
from functools import cached_property
from itertools import chain, combinations
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


class CKBBJoint(BaseModel):
    model_config = ConfigDict(frozen=True)
    check_type: PauliChar
    stack: List[TannerNode]


class CKBBAncillaTanner(TannerGraph):
    port: Dict[TannerNode, TannerNode]
    joint: List[CKBBJoint]

    def __or__(self, other: "CKBBAncillaTanner") -> "CKBBAncillaTanner":
        new_variable_nodes = self.variable_nodes | other.variable_nodes
        new_check_nodes = self.check_nodes | other.check_nodes
        new_edges = self.edges | other.edges
        new_port = {**self.port, **other.port}
        new_joint = self.joint + other.joint
        return CKBBAncillaTanner(
            variable_nodes=new_variable_nodes,
            check_nodes=new_check_nodes,
            edges=new_edges,
            port=new_port,
            joint=new_joint,
        )


class CKBBMeasurement(PauliMeasurement):

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
        ancilla_tanners = {}
        for lop, support in lop_tanners.items():
            ancilla_tanners[lop] = self._build_ancilla_tanner(support)

        # Order to optimize cost.

        # III. Build full ancilla by joining ancillas tanners
        disjoint_support = list(ancilla_tanners.keys())
        # First element
        prev_key = disjoint_support[0]
        full_ancilla_tanner = ancilla_tanners[prev_key].copy()

        # Link all follwing element with the previous one and build the tanner graph.
        if len(ancilla_tanners) > 1:
            for lop in disjoint_support[1:]:
                next_tanner = ancilla_tanners[lop]
                if len(ancilla_tanners[prev_key].joint) < 2:
                    raise ValueError(
                        "Error in building measurement: ancilla Tanner has less than 2 joints, cannot build bridge to the next ancilla Tanner."
                    )
                bridge_tanner, linking_edges = self._build_bridge_tanner(
                    (ancilla_tanners[prev_key].joint[1], next_tanner.joint[0])
                )

                full_ancilla_tanner |= next_tanner
                full_ancilla_tanner = tga.connect(
                    full_ancilla_tanner, bridge_tanner, connecting_edges=linking_edges
                )

                prev_key = lop

        # IV. Build merged Tanner

        distinct_code = (
            set()
        )  # identify a set of all distinct codes involved in the measurement.
        tanner_codes = TannerGraph()  # Sum of tanner of codes involved (disconnected)
        for lop in self.logical_targets:
            # if isinstance(lop, tuple):
            #     # All element of the tuple should have the same code assignement, if not it would mean a dataqubit is shared between two codes.
            #     code = self.context.initial_assignement[lop[0]]
            lop_code = self.context.initial_assignement[lop]
            if lop_code.id not in distinct_code:
                tanner_codes |= lop_code.tanner_graph
            distinct_code.add(lop_code.id)

        connecting_edges = []
        check_anticommuting_with_lop = defaultdict(set)

        for lop in self.logical_targets:
            code = self.context.initial_assignement[lop]
            # construct edges connecting the ancilla Tanner to the code Tanner.
            port = ancilla_tanners[lop].port
            for p1, p2 in port.items():
                if (
                    p1 in code.tanner_graph.variable_nodes
                    and p2 in full_ancilla_tanner.check_nodes
                ):
                    # Check if any element of the opposite logical operator is measured by the ancilla system's stabilisers.
                    for lqb in code.logical_qubits:
                        if p2.check_type == CheckNode.CheckType.X:
                            if p1 in lqb.logical_z.target_nodes:
                                print(
                                    f"Check {p2} anticommutes with logical operator {lqb.logical_z} because of node {p1}"
                                )
                                check_anticommuting_with_lop[lqb.logical_z].add(p2)
                        if p2.check_type == CheckNode.CheckType.Z:
                            if p1 in lqb.logical_x.target_nodes:
                                print(
                                    f"Check {p2} anticommutes with logical operator {lqb.logical_x} because of node {p1}"
                                )
                                check_anticommuting_with_lop[lqb.logical_x].add(p2)
                    connecting_edges.append(
                        TannerEdge(
                            variable_node=p1,
                            check_node=p2,
                            pauli_checked=p2.check_type,
                        )
                    )
                elif (
                    p1 in code.tanner_graph.check_nodes
                    and p2 in full_ancilla_tanner.variable_nodes
                ):
                    connecting_edges.append(
                        TannerEdge(
                            variable_node=p2,
                            check_node=p1,
                            pauli_checked=p1.check_type,
                        )
                    )
                else:
                    raise ValueError(
                        "Error in building merged code: port mapping is inconsistent."
                    )

        merged_tanner = tga.connect(
            full_ancilla_tanner, tanner_codes, connecting_edges=connecting_edges
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
                target_nodes=full_ancilla_tanner.variable_nodes,
                gates=var_node_initial_state.pauli_from_zero(),
            ),
        ]
        observable_nodes = {
            "XX_outcome": set(
                [
                    n
                    for n in full_ancilla_tanner.check_nodes
                    if n.check_type == check_type
                ]
            ),
        }
        for n in full_ancilla_tanner.check_nodes:
            observable_nodes[f"ckbb_check_{n.tag}_{n.check_type}"] = {n}

        i = 1
        for k, v in check_anticommuting_with_lop.items():
            print(f"anticommute_with_{i}: {[n.tag for n in v]}")
            observable_nodes[f"anticommute_with_{i}"] = v
            i += 1

        stab_measurement = StabilisersMeasurementCompiler(
            data=merged_tanner,
            round=self.distance,
            tag=f"ckbb_stab_{self.tag}",
            observable_included=observable_nodes,
        )

        readout_ancilla = []
        for v in full_ancilla_tanner.variable_nodes:
            readout_ancilla.append(
                MeasurementCompiler(
                    data=TannerGraph(
                        variable_nodes={v}, check_nodes=set(), edges=set()
                    ),
                    tag=f"ckbb_var_{v.tag}",
                    basis=basis,
                    reset_qubits=False,
                    free_qubits=False,
                )
            )
        compilers.extend(init_ancilla)
        compilers.append(stab_measurement)
        compilers.extend(readout_ancilla)
        return compilers

    def _build_ancilla_tanner(self, support: TannerGraph) -> CKBBAncillaTanner:
        # Build the ancilla TannerGraph for a given logical operator support
        # Identify ports and joints

        # Layer 1 is the dual of the support, its qubits are the "port", i.e. the connection points between the ancilla and the code TannerGraph.
        # map_between is a map from node id of the support to those of the port.
        bottom_layer, map_between = tga.dual_graph(support)

        # indexing nodes in order to keep track of the mapping between layers, and identify joints.
        prev_idx = tga.index_nodes(bottom_layer)
        prev_tanner = bottom_layer.copy()

        layers = [prev_idx]
        ancilla_tanner = prev_tanner

        # add 2*r-2 layers interleaving the dual and primal of the support TannerGraph, and connect them with edges between corresponding nodes. Identify joints along the way.
        for r in range(2 * self.distance - 2):
            v, vidx = tga.indexed_dual_graph(prev_tanner, prev_idx)
            inter_layer_edges = []

            for n_idx, node in prev_idx.items():
                if node in prev_tanner.check_nodes and vidx[n_idx] in v.variable_nodes:
                    inter_layer_edges.append(
                        TannerEdge(
                            variable_node=vidx[n_idx],
                            check_node=prev_idx[n_idx],
                            pauli_checked=prev_idx[n_idx].check_type,
                        )
                    )
                elif (
                    node in prev_tanner.variable_nodes and vidx[n_idx] in v.check_nodes
                ):
                    inter_layer_edges.append(
                        TannerEdge(
                            variable_node=prev_idx[n_idx],
                            check_node=vidx[n_idx],
                            pauli_checked=vidx[n_idx].check_type,
                        )
                    )
                else:
                    raise ValueError(
                        "Error in building ancilla Tanner: node mapping is inconsistent."
                    )
            ancilla_tanner = tga.connect(ancilla_tanner, v, inter_layer_edges)
            layers.append(vidx)
            prev_tanner = v
            prev_idx = vidx

        stacks = [[] for _ in range(bottom_layer.number_of_nodes)]
        for l in layers:
            for bl in range(bottom_layer.number_of_nodes):
                stacks[bl].append(l[bl])

        joints = []
        for s in stacks:
            if s[0] in bottom_layer.check_nodes:
                joints.append(CKBBJoint(check_type=s[0].check_type, stack=s))
                # TODO Can we use vertical line based on data qubits as joint ?

        return CKBBAncillaTanner(
            variable_nodes=ancilla_tanner.variable_nodes,
            check_nodes=ancilla_tanner.check_nodes,
            edges=ancilla_tanner.edges,
            port=map_between,
            joint=joints,
        )

    def _build_bridge_tanner(
        self, bridge_ends: tuple[CKBBJoint, CKBBJoint]
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
