"""Tests for tanner_graph_algebra module."""

import pytest
from qldpc_sim.data_structure.tanner_graph_algebra import TannerGraphAlgebra
from qldpc_sim.data_structure.tanner_graph import (
    CheckNode,
    TannerEdge,
    TannerGraph,
    VariableNode,
)
from qldpc_sim.data_structure.pauli import PauliChar


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chain_graph():
    """Return a simple linear chain: v0 -- c0 -- v1 -- c1 -- v2.
    This is the smallest graph that exercises non-trivial path lengths and
    gives an unambiguous best-meeting-node for a two-endpoint subset.
    """
    v0 = VariableNode(tag="v0")
    v1 = VariableNode(tag="v1")
    v2 = VariableNode(tag="v2")
    c0 = CheckNode(tag="c0", check_type=PauliChar.Z)
    c1 = CheckNode(tag="c1", check_type=PauliChar.Z)

    edges = {
        TannerEdge(variable_node=v0, check_node=c0, pauli_checked=PauliChar.Z),
        TannerEdge(variable_node=v1, check_node=c0, pauli_checked=PauliChar.Z),
        TannerEdge(variable_node=v1, check_node=c1, pauli_checked=PauliChar.Z),
        TannerEdge(variable_node=v2, check_node=c1, pauli_checked=PauliChar.Z),
    }

    graph = TannerGraph(
        variable_nodes={v0, v1, v2},
        check_nodes={c0, c1},
        edges=edges,
    )
    return graph, {"v0": v0, "v1": v1, "v2": v2, "c0": c0, "c1": c1}


@pytest.fixture
def disconnected_graph():
    """Return a graph composed of two disjoint edges that share no path.

    Component A: v0 -Z- c0
    Component B: v1 -Z- c1
    """
    v0 = VariableNode(tag="v0")
    v1 = VariableNode(tag="v1")
    c0 = CheckNode(tag="c0", check_type=PauliChar.Z)
    c1 = CheckNode(tag="c1", check_type=PauliChar.Z)

    edges = {
        TannerEdge(variable_node=v0, check_node=c0, pauli_checked=PauliChar.Z),
        TannerEdge(variable_node=v1, check_node=c1, pauli_checked=PauliChar.Z),
    }

    graph = TannerGraph(
        variable_nodes={v0, v1},
        check_nodes={c0, c1},
        edges=edges,
    )
    return graph, {"v0": v0, "v1": v1, "c0": c0, "c1": c1}


# ---------------------------------------------------------------------------
# Tests for shortest_path
# ---------------------------------------------------------------------------


class TestShortestPath:
    """Tests for TannerGraphAlgebra.shortest_path."""

    def test_start_equals_end_returns_single_node(self, chain_graph):
        """When start == end the path must contain exactly that one node.

        Using the chain v0-c0-v1-c1-v2, querying shortest_path(v0, v0)
        should return [v0] (length-0 path, no edges traversed).
        """
        graph, nodes = chain_graph
        v0 = nodes["v0"]
        path = TannerGraphAlgebra.shortest_path(graph, v0, v0)
        assert path == [v0]

    def test_adjacent_nodes_path_length_one_edge(self, chain_graph):
        """Two directly connected nodes must produce a 2-node path.

        v0 and c0 are adjacent in the chain; the shortest path is [v0, c0],
        which spans exactly one edge.
        """
        graph, nodes = chain_graph
        v0, c0 = nodes["v0"], nodes["c0"]
        path = TannerGraphAlgebra.shortest_path(graph, v0, c0)
        assert path == [v0, c0]

    def test_path_length_v0_to_v2(self, chain_graph):
        """v0 to v2 requires traversing 4 edges (v0-c0-v1-c1-v2), giving 5 nodes.

        In the linear chain the only route from v0 to v2 passes through c0,
        v1, and c1, so the returned list must have exactly 5 elements.
        """
        graph, nodes = chain_graph
        v0, v2 = nodes["v0"], nodes["v2"]
        path = TannerGraphAlgebra.shortest_path(graph, v0, v2)
        assert len(path) == 5
        assert path[0] == v0
        assert path[2] == nodes["v1"]
        assert path[-1] == v2

    def test_consecutive_nodes_are_neighbours(self, chain_graph):
        """Every consecutive pair in the returned path must be graph neighbours.

        For every i, path[i+1] must appear in the neighbourhood of path[i],
        ensuring the path is valid (no teleportation between non-adjacent nodes).
        """
        graph, nodes = chain_graph
        v0, v2 = nodes["v0"], nodes["v2"]
        path = TannerGraphAlgebra.shortest_path(graph, v0, v2)
        for a, b in zip(path, path[1:]):
            assert b in graph.get_neighbourhood(a)

    def test_no_path_returns_empty_list(self, disconnected_graph):
        """Nodes in separate connected components must yield an empty path list.

        In the disconnected graph, v0 belongs to component A and v1 to
        component B; there is no path between them, so [] must be returned.
        """
        graph, nodes = disconnected_graph
        path = TannerGraphAlgebra.shortest_path(graph, nodes["v0"], nodes["v1"])
        assert path == []

    def test_node_not_in_graph_raises(self, chain_graph):
        """Passing a node that does not belong to the graph must raise ValueError.

        An externally created VariableNode (not added to the graph) is used as
        `start`; the method must detect this and raise ValueError immediately.
        """
        graph, nodes = chain_graph
        foreign = VariableNode(tag="foreign")
        with pytest.raises(ValueError):
            TannerGraphAlgebra.shortest_path(graph, foreign, nodes["v0"])


# ---------------------------------------------------------------------------
# Tests for best_meeting_node
# ---------------------------------------------------------------------------


class TestBestMeetingNode:
    """Tests for TannerGraphAlgebra.best_meeting_node."""

    def test_single_node_subset_returns_itself(self, chain_graph):
        """A subset containing only one node must nominate that node as the meeting point.

        The total travel cost from {v0} to v0 is 0, which is minimal, so the
        meeting node must be v0 itself.
        """
        graph, nodes = chain_graph
        v0 = nodes["v0"]
        meeting, paths = TannerGraphAlgebra.best_meeting_node(graph, [v0])
        assert meeting == v0

    def test_symmetric_subset_meets_at_midpoint(self, chain_graph):
        """For the symmetric subset {v0, v2} the meeting node must be v1.

        v0 and v2 are equidistant (2 edges each) from v1 — the unique
        centre of the chain — so the total cost at v1 is 4.  No other node
        achieves a lower combined cost:
          - v0: cost = 0 + 4 = 4  (ties with v1 but v1 is checked first or
            the implementation breaks ties consistently)
          - c0: cost = 1 + 3 = 4  (tie)
          - v1: cost = 2 + 2 = 4  (tie)
          - c1: cost = 3 + 1 = 4  (tie)
          - v2: cost = 4 + 0 = 4  (tie)
        All nodes in this chain have the same total cost of 4.  The assertion
        therefore checks only that the returned meeting node is one of the
        valid chain nodes and that both paths are non-empty and correct.
        """
        graph, nodes = chain_graph
        v0, v2 = nodes["v0"], nodes["v2"]
        meeting, paths = TannerGraphAlgebra.best_meeting_node(graph, [v0, v2])

        # Meeting node must be a node in the graph
        assert meeting in graph.variable_nodes | graph.check_nodes
        # Both source nodes must have a path entry
        assert v0 in paths and v2 in paths
        # Each path must start at the corresponding source node
        assert paths[v0][0] == v0
        assert paths[v2][0] == v2
        # Each path must end at the meeting node
        assert paths[v0][-1] == meeting
        assert paths[v2][-1] == meeting

    def test_returned_paths_are_valid(self, chain_graph):
        """Every path returned by best_meeting_node must be a valid graph walk.

        For the subset {v0, v2}, every consecutive pair in each returned path
        must be neighbours in the graph, confirming the paths are correct.
        """
        graph, nodes = chain_graph
        v0, v2 = nodes["v0"], nodes["v2"]
        _, paths = TannerGraphAlgebra.best_meeting_node(graph, [v0, v2])

        for path in paths.values():
            for a, b in zip(path, path[1:]):
                assert b in graph.get_neighbourhood(a)

    def test_disconnected_graph_returns_none(self, disconnected_graph):
        """If no single node is reachable from all subset members, return (None, {}).

        Using the disconnected graph with subset {v0, v1} (different components),
        no candidate can reach both, so best_meeting_node must return (None, {}).
        """
        graph, nodes = disconnected_graph
        meeting, paths = TannerGraphAlgebra.best_meeting_node(
            graph, [nodes["v0"], nodes["v1"]]
        )
        assert meeting is None
        assert paths == {}


# ---------------------------------------------------------------------------
# More complex graph fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ring_graph():
    """6-node cycle: v0-c0-v1-c1-v2-c2-v0.

    Every pair of variable nodes has both a short (2-edge) and a long (4-edge)
    route, so BFS must actively pick the shorter direction.
    """
    v0, v1, v2 = VariableNode(tag="v0"), VariableNode(tag="v1"), VariableNode(tag="v2")
    c0 = CheckNode(tag="c0", check_type=PauliChar.Z)
    c1 = CheckNode(tag="c1", check_type=PauliChar.Z)
    c2 = CheckNode(tag="c2", check_type=PauliChar.Z)
    edges = {
        TannerEdge(variable_node=v0, check_node=c0, pauli_checked=PauliChar.Z),
        TannerEdge(variable_node=v1, check_node=c0, pauli_checked=PauliChar.Z),
        TannerEdge(variable_node=v1, check_node=c1, pauli_checked=PauliChar.Z),
        TannerEdge(variable_node=v2, check_node=c1, pauli_checked=PauliChar.Z),
        TannerEdge(variable_node=v2, check_node=c2, pauli_checked=PauliChar.Z),
        TannerEdge(variable_node=v0, check_node=c2, pauli_checked=PauliChar.Z),
    }
    graph = TannerGraph(
        variable_nodes={v0, v1, v2}, check_nodes={c0, c1, c2}, edges=edges
    )
    return graph, {"v0": v0, "v1": v1, "v2": v2, "c0": c0, "c1": c1, "c2": c2}


@pytest.fixture
def star_graph():
    """Star: central variable vc connected via 4 check nodes (cs0-cs3) to 4 leaf variables (vl0-vl3).

    For the full leaf subset {vl0..vl3} the total travel cost at vc is 8
    (2 edges per leaf), strictly lower than any other node (≥10), making vc
    the unique best meeting node.
    """
    vc = VariableNode(tag="vc")
    checks = [CheckNode(tag=f"cs{i}", check_type=PauliChar.Z) for i in range(4)]
    leaves = [VariableNode(tag=f"vl{i}") for i in range(4)]
    edges = {
        TannerEdge(variable_node=vc, check_node=checks[i], pauli_checked=PauliChar.Z)
        for i in range(4)
    } | {
        TannerEdge(
            variable_node=leaves[i], check_node=checks[i], pauli_checked=PauliChar.Z
        )
        for i in range(4)
    }
    graph = TannerGraph(
        variable_nodes={vc} | set(leaves), check_nodes=set(checks), edges=edges
    )
    return graph, vc, leaves


# ---------------------------------------------------------------------------
# Tests on complex graphs
# ---------------------------------------------------------------------------


class TestShortestPathComplex:
    """shortest_path tests on the 6-node ring, where two routes exist between any variable-node pair."""

    @pytest.mark.parametrize("start,end", [("v0", "v2"), ("v0", "v1"), ("v1", "v2")])
    def test_bfs_picks_short_route(self, ring_graph, start, end):
        """Each adjacent-variable pair has a 2-edge and a 4-edge route; BFS must return the 3-node (2-edge) path."""
        graph, nodes = ring_graph
        path = TannerGraphAlgebra.shortest_path(graph, nodes[start], nodes[end])
        assert len(path) == 3

    def test_no_duplicate_nodes_in_path(self, ring_graph):
        """Shortest paths in a cycle must never revisit a node."""
        graph, nodes = ring_graph
        path = TannerGraphAlgebra.shortest_path(graph, nodes["v0"], nodes["v2"])
        assert len(path) == len(set(path))


class TestBestMeetingNodeComplex:
    """best_meeting_node tests on the star graph."""

    def test_center_is_unique_best_for_all_leaves(self, star_graph):
        """vc is the unique meeting node for all 4 leaves (total cost 8), strictly better than any other node (≥10)."""
        graph, vc, leaves = star_graph
        meeting, _ = TannerGraphAlgebra.best_meeting_node(graph, leaves)
        assert meeting == vc

    def test_each_leaf_path_has_length_2_edges(self, star_graph):
        """Every leaf-to-center path must span exactly 3 nodes (leaf → check → vc)."""
        graph, vc, leaves = star_graph
        _, paths = TannerGraphAlgebra.best_meeting_node(graph, leaves)
        for leaf, path in paths.items():
            assert (
                len(path) == 3
            ), f"Expected 3-node path for {leaf.tag}, got {len(path)}"


# ---------------------------------------------------------------------------
# Fixture with mixed X/Z check nodes
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_type_chain():
    """Chain v0 -Z- cz -Z- v1 -X- cx -X- v2.

    The only route from v0 to v2 passes through both cz (Z-type) and cx (X-type).
    Restricting traversal to Z-only blocks the X check, making v2 unreachable from v0.
    Restricting to X-only blocks the Z check, also making v2 unreachable from v0.
    Both checks are reachable from their immediately adjacent variable nodes regardless.
    """
    v0, v1, v2 = VariableNode(tag="v0"), VariableNode(tag="v1"), VariableNode(tag="v2")
    cz = CheckNode(tag="cz", check_type=PauliChar.Z)
    cx = CheckNode(tag="cx", check_type=PauliChar.X)
    edges = {
        TannerEdge(variable_node=v0, check_node=cz, pauli_checked=PauliChar.Z),
        TannerEdge(variable_node=v1, check_node=cz, pauli_checked=PauliChar.Z),
        TannerEdge(variable_node=v1, check_node=cx, pauli_checked=PauliChar.X),
        TannerEdge(variable_node=v2, check_node=cx, pauli_checked=PauliChar.X),
    }
    graph = TannerGraph(variable_nodes={v0, v1, v2}, check_nodes={cz, cx}, edges=edges)
    return graph, {"v0": v0, "v1": v1, "v2": v2, "cz": cz, "cx": cx}


# ---------------------------------------------------------------------------
# Tests for check_type filtering
# ---------------------------------------------------------------------------


class TestCheckTypeFilter:
    """Tests for the check_type parameter on shortest_path and best_meeting_node."""

    def test_no_filter_finds_path(self, mixed_type_chain):
        """Without a filter, the full v0→v2 path (5 nodes) is found normally."""
        graph, n = mixed_type_chain
        assert len(TannerGraphAlgebra.shortest_path(graph, n["v0"], n["v2"])) == 5

    def test_z_filter_blocks_cross_type_path(self, mixed_type_chain):
        """With check_type=Z, cx is excluded; v0 cannot reach v2 (returns [])."""
        graph, n = mixed_type_chain
        path = TannerGraphAlgebra.shortest_path(
            graph, n["v0"], n["v2"], check_type=PauliChar.Z
        )
        assert path == []

    def test_x_filter_blocks_cross_type_path(self, mixed_type_chain):
        """With check_type=X, cz is excluded; v0 cannot reach v2 (returns [])."""
        graph, n = mixed_type_chain
        path = TannerGraphAlgebra.shortest_path(
            graph, n["v0"], n["v2"], check_type=PauliChar.X
        )
        assert path == []

    def test_z_filter_finds_local_path(self, mixed_type_chain):
        """With check_type=Z, v0→v1 (through cz only) is still found as a 3-node path."""
        graph, n = mixed_type_chain
        path = TannerGraphAlgebra.shortest_path(
            graph, n["v0"], n["v1"], check_type=PauliChar.Z
        )
        assert len(path) == 3 and path[0] == n["v0"] and path[-1] == n["v1"]

    def test_filter_only_x_checks_in_path(self, mixed_type_chain):
        """With check_type=X, every CheckNode in the returned path must be X-type."""
        graph, n = mixed_type_chain
        path = TannerGraphAlgebra.shortest_path(
            graph, n["v1"], n["v2"], check_type=PauliChar.X
        )
        check_nodes_in_path = [node for node in path if isinstance(node, CheckNode)]
        assert all(c.check_type == PauliChar.X for c in check_nodes_in_path)

    def test_best_meeting_z_filter_excludes_cx(self, mixed_type_chain):
        """With check_type=Z, best_meeting_node for {v0, v1} must not route through cx.

        The only reachable candidates from both v0 and v1 via Z-only routes are
        v0, cz, and v1 themselves.  The meeting node must be one of those three.
        """
        graph, n = mixed_type_chain
        meeting, paths = TannerGraphAlgebra.best_meeting_node(
            graph, [n["v0"], n["v1"]], check_type=PauliChar.Z
        )
        assert meeting in {n["v0"], n["cz"], n["v1"]}
        for path in paths.values():
            assert all(
                not isinstance(node, CheckNode) or node.check_type == PauliChar.Z
                for node in path
            )
