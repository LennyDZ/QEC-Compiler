from collections import defaultdict
from functools import cached_property
from itertools import chain, combinations
from typing import Dict, List, Set, Tuple
from pydantic import BaseModel, ConfigDict

from qldpc_sim.qldpc_experiment.record import EventType, OutcomeSet

from ..data_structure import (
    TannerGraph,
    LogicalOperator,
    CheckNode,
    TannerEdge,
    TannerNode,
    VariableNode,
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
    """An extension of tanner graph to describe CKBB ancilla system.
    This add property to describe the following:
    - port: the mapping between the ancilla TannerGraph and the code TannerGraph, i.e. which node of the ancilla TannerGraph is connected to which node of the code TannerGraph. This is necessary to keep track of how the ancilla system is connected to the code, and to build the final merged TannerGraph.
    - joint: a list of CKBBJoint objects available, these are used to bridge different ancilla TannerGraph together when the measurement support more than one logical operator. Each joint is a set of nodes across layers of the ancilla TannerGraph that can be measured together to perform the bridge operation. The check type of the joint is also specified, as it determine how the bridge is performed.
    """

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
        """Build tanner graph of the support of all logical operators involved in the measurement.
        CKBB only works when the support of the logical operators are disjoint, so the tanner graph of the support of each logical operator can be built independently.
        """
        tanners = {}
        for lop in self.logical_targets:
            code = self.context.initial_assignement[lop]
            new_support = code.tanner_graph.get_support(
                lop.target_nodes, lop.logical_type
            )
            new_key = lop
            for key, other_support in tanners.items():
                if not new_support.is_disjoint(other_support):
                    raise ValueError(
                        "Error in building measurement: two logical operators have overlapping support"
                    )

            tanners[new_key] = new_support
        return tanners

    def check_feasibility(self):
        # Implement feasibility checks specific to Measurement
        # Check if operators are disjoint or not, etc.
        return True

    def cost(self):
        """Compute the number of ancilla qubits required for the measurement."""
        cost = 0
        for lop, support in self.tanner_supports.items():
            cost += support.number_of_nodes
        cost *= 2 * self.distance - 2  # number of layers in the ancilla TannerGraph
        return cost

    def build_compiler_instructions(self):
        compilers = []
        # I. Evalute feasability, cost
        if not self.check_feasibility():
            raise ValueError(
                "Measurement instruction is not feasible with the given memory."
            )

        # TODO: one may assume we could allow operator with mixed X and Z type. In this case, we need more complicated compiler instruction as some part of the ancilla must be init in X and some part in Z. For now we only allow measurement of parity involving only X or only Z operator.
        basis = self.logical_targets[0].logical_type
        for lop in self.logical_targets:
            if lop.logical_type != basis:
                raise ValueError(
                    "Error in building measurement: logical operators have different type, this is not supported for now."
                )

        # Set the initial state of the ancilla
        if basis == PauliChar.X:
            check_type = PauliChar.X
            var_node_initial_state = "RZ"
        elif basis == PauliChar.Z:
            check_type = PauliChar.Z
            var_node_initial_state = "RX"
        else:
            raise ValueError("Only X and Z measurement are supported for now.")

        # II. Build ancilla TannerGraphs
        lop_tanners = self.tanner_supports
        ancilla_tanners = {}
        for lop, support in lop_tanners.items():
            # for each logical operator, build a CKBBAncillaTanner.
            ancilla_tanners[lop] = self._build_ancilla_tanner(support)

        # TODO: Order to optimize cost ? (added this earlier, not sure anymore how it can be improved)

        # III. Build full ancilla by joining ancillas tanners
        list_of_anc_tanner = list(ancilla_tanners.keys())
        # First element
        prev_key = list_of_anc_tanner[0]
        full_ancilla_tanner = ancilla_tanners[prev_key].copy()

        # Link all follwing element with the previous one and build the tanner graph.
        # joint using port 0 of current to port 1 of previous.
        if len(ancilla_tanners) > 1:
            for lop in list_of_anc_tanner[1:]:
                next_tanner = ancilla_tanners[lop]
                if len(ancilla_tanners[prev_key].joint) < 2:
                    # TODO: improve this by allowing at most 2 lop with only 1 joint (putting them at the beginning and the end of the line)
                    raise ValueError(
                        "Error in building measurement: ancilla Tanner has less than 2 joints, cannot build bridge to the next ancilla Tanner."
                    )
                # Build bridge between prev and next.
                bridge_tanner, linking_edges = self._build_bridge_tanner(
                    (ancilla_tanners[prev_key].joint[1], next_tanner.joint[0])
                )

                # First merge the 2 (disjoint) tanner graph by simple union
                # Then connect the bridge tanner using the linking edges (which include "both sides")
                full_ancilla_tanner |= next_tanner
                full_ancilla_tanner = tga.connect(
                    full_ancilla_tanner, bridge_tanner, connecting_edges=linking_edges
                )

                prev_key = lop

        # IV. Build merged Tanner
        distinct_code = (
            set()
        )  # identify a set of all distinct codes involved in the measurement.
        tanner_codes = (
            TannerGraph()
        )  # Sum of disjoint tanner of codes involved (disconnected)
        for lop in self.logical_targets:
            lop_code = self.context.initial_assignement[lop]
            if lop_code.id not in distinct_code:
                tanner_codes |= lop_code.tanner_graph
                distinct_code.add(lop_code.id)

        connecting_edges = []
        for lop in self.logical_targets:
            code = self.context.initial_assignement[lop]
            # construct edges connecting the ancilla Tanner to the code Tanner.
            port = ancilla_tanners[lop].port
            for p1, p2 in port.items():
                # Case 1, check node in ancilla, data node in code
                if (
                    p1 in code.tanner_graph.variable_nodes
                    and p2 in full_ancilla_tanner.check_nodes
                ):
                    connecting_edges.append(
                        TannerEdge(
                            variable_node=p1,
                            check_node=p2,
                            pauli_checked=p2.check_type,
                        )
                    )
                # Case 2, check node in code, data node in ancilla
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

        # Compute classical correction of logicals that commute with the supports used.
        anticommuting_lop_by_node = {}
        corrections = {}
        for lop in self.logical_targets:
            code = self.context.initial_assignement[lop]
            for lq in code.logical_qubits:
                if lq.logical_x == lop or lq.logical_z == lop:
                    shared_target_nodes = set(lq.logical_x.target_nodes) & set(
                        lq.logical_z.target_nodes
                    )
                    if len(shared_target_nodes) > 1:
                        raise ValueError(
                            "Error in building measurement: a logical operator share more than 1 qubit with its anticommuting logical operator, this is not supported for now. Try finding a canonical basis for the code."
                        )
                    anticommuting_lop_by_node[shared_target_nodes.pop()] = (
                        lq.logical_x if lq.logical_z == lop else lq.logical_z
                    )
                    break

        meeting, paths = tga.best_meeting_node(
            merged_tanner, anticommuting_lop_by_node.keys(), check_type=basis
        )

        # Keep the shared meeting node in exactly one correction path so bridge
        # information is retained once, while avoiding duplicate counting.
        meeting_keeper = None
        for node in anticommuting_lop_by_node.keys():
            if meeting in paths[node]:
                meeting_keeper = node
                break

        for node, lop in anticommuting_lop_by_node.items():
            path_nodes = list(paths[node])
            if meeting in path_nodes and node != meeting_keeper:
                path_nodes.remove(meeting)

            corrections[lop] = {n for n in path_nodes if isinstance(n, VariableNode)}

        # V. Build compiler

        init_ancilla = [
            ApplyGates(
                tag=f"init_{self.tag}",
                target_nodes=full_ancilla_tanner.variable_nodes,
                gates=[var_node_initial_state],
            ),
        ]

        stab_measurement = StabilisersMeasurementCompiler(
            data=merged_tanner,
            round=self.distance,
            tag=f"ckbb_{self.tag}",
        )

        # Readout the ancilla, providing the correction information for any logical affected.
        readout_ancilla = MeasurementCompiler(
            data=TannerGraph(
                variable_nodes=full_ancilla_tanner.variable_nodes,
                check_nodes=set(),
                edges=set(),
            ),
            tag=f"ckbb_{self.tag}_anc_readout",
            basis=basis.dual(),
            reset_qubits=True,
            free_qubits=True,
        )
        compilers.extend(init_ancilla)
        compilers.append(stab_measurement)
        compilers.append(readout_ancilla)

        # Nodes measured by this gadget: stabiliser checks and ancilla readout variables.
        measured_nodes_in_gadget = set(merged_tanner.check_nodes) | set(
            full_ancilla_tanner.variable_nodes
        )

        outcomes = []
        stab_in_ancilla = set(
            [n for n in full_ancilla_tanner.check_nodes if n.check_type == check_type]
        )
        parity_outcome_nodes = OutcomeSet(
            tag=f"{self.tag}_parity_outcome",
            type=EventType.OBSERVABLE,
            size=len(stab_in_ancilla),
            measured_nodes=stab_in_ancilla,
            target=self.logical_targets,
        )
        outcomes.append(parity_outcome_nodes)
        # Logical correction :
        for lop, cond_nodes in corrections.items():
            correction_nodes = set(cond_nodes) & measured_nodes_in_gadget
            if not correction_nodes:
                continue
            outcomes.append(
                OutcomeSet(
                    tag=f"{self.tag}log_corr_{lop.id}",
                    type=EventType.FRAME_CORRECTION,
                    size=len(correction_nodes),
                    measured_nodes=correction_nodes,
                    target=lop,
                )
            )

        return compilers, outcomes

    def _build_ancilla_tanner(
        self, support: TannerGraph, system_coord: int = None
    ) -> CKBBAncillaTanner:
        """Build the layered ancilla tanner for a given logical support (support is a subset of a code)"""

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
        """Build the bridge between to CKBBJoint. Only allow joint between XX or ZZ.

        Return the tanner graph of the bridge, and the set of edges connecting the bridge to the 2 joint.
        """
        connecting_edges = set()
        bridge_edges = set()
        var_nodes = set()
        check_nodes = set()
        # joint are of the same type (easy case)
        if bridge_ends[0].check_type != bridge_ends[1].check_type:
            raise ValueError(
                "Error in building bridge Tanner: only joint between same check type are supported for now."
            )

        c_type = bridge_ends[0].check_type
        prev_layer_node = None
        # iter nodes connected to the bridge (at each layer)
        for i, (s1, s2) in enumerate(zip(bridge_ends[0].stack, bridge_ends[1].stack)):
            # Add a check between two variable nodes.
            if isinstance(s1, VariableNode) and isinstance(s2, VariableNode):
                check_type = c_type.dual()
                new_check = CheckNode(
                    tag=f"bc_l{i}_{check_type}",
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
                    # Add edge between bridges nodes
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
                    tag=f"bv_l{i}",
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

        return (
            TannerGraph(
                variable_nodes=var_nodes, check_nodes=check_nodes, edges=bridge_edges
            ),
            connecting_edges,
        )
