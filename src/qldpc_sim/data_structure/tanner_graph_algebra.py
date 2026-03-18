import random
from collections import deque
from typing import Dict, List, Tuple
from .pauli import PauliChar
from .tanner_graph import CheckNode, TannerEdge, TannerGraph, TannerNode, VariableNode


class TannerGraphAlgebra:
    """This class provides utility function to work with tanner graphs."""

    def connect(
        graph1: TannerGraph, graph2: TannerGraph, connecting_edges: List[TannerEdge]
    ) -> TannerGraph:
        """Connect two tanner graph into a new Tanner graph, using the connecting edges provided.

        Parameters
        ----------
        graph1 : TannerGraph
        graph2 : TannerGraph
        connecting_edges : List[TannerEdge]

        Returns
        -------
        TannerGraph

        Raises
        ------
        ValueError
            If the two graphs have overlapping nodes, or if the connecting edges do not connect nodes from different graphs.
        ValueError
            If the connecting edges do not connect nodes from different graphs.
        """
        # Implementation of merging two Tanner graphs
        if not graph1.variable_nodes.isdisjoint(
            graph2.variable_nodes
        ) or not graph1.check_nodes.isdisjoint(graph2.check_nodes):
            raise ValueError("Graphs have overlapping nodes; cannot merge.")

        for ce in connecting_edges:
            valid_edge = (
                ce.variable_node in graph1.variable_nodes
                and ce.check_node in graph2.check_nodes
                or ce.variable_node in graph2.variable_nodes
                and ce.check_node in graph1.check_nodes
            )
            if not valid_edge:
                raise ValueError(
                    "Connecting edge does not connect nodes from different graphs."
                )
        merged_variable_nodes = graph1.variable_nodes.union(graph2.variable_nodes)
        merged_check_nodes = graph1.check_nodes.union(graph2.check_nodes)
        merged_edges = graph1.edges.union(graph2.edges).union(set(connecting_edges))
        return TannerGraph(
            variable_nodes=merged_variable_nodes,
            check_nodes=merged_check_nodes,
            edges=merged_edges,
        )

    def dual_graph(
        graph: TannerGraph, system_coord: int = None
    ) -> Tuple[TannerGraph, Dict[TannerNode, TannerNode]]:
        """Return a dual of the Tanner graph. A dual graph is constructed by swapping the variable and check nodes, and connecting them according to the original edges. The Pauli checked on the edges are also dualized (X<->Z, Y->Y).

        Parameters
        ----------
        graph : TannerGraph
            The Tanner graph to be dualized.
        system_coord : int, optional
            The system coordinate for the dual graph, by default None.

        Returns
        -------
        Tuple[TannerGraph, Dict[TannerNode, TannerNode]]
            The dual Tanner graph, and a mapping from the original nodes to the "corresponding" new nodes in the dual graph.
        """

        def dual_tag(tag: str) -> str:
            t = ""
            if tag.endswith("_T"):
                t += tag[:-2]
            else:
                t += tag + "_T"
            return "_" + t

        check_node = random.choice(tuple(graph.check_nodes))
        check_type = check_node.check_type.dual()

        check_to_variable = {
            check: VariableNode(tag=dual_tag(check.tag)) for check in graph.check_nodes
        }
        variable_to_check = {
            variable: CheckNode(
                tag=dual_tag(variable.tag),
                check_type=check_type,
            )
            for variable in graph.variable_nodes
        }

        dual_variable_nodes = set(check_to_variable.values())
        dual_check_nodes = set(variable_to_check.values())

        dual_edges = {
            TannerEdge(
                variable_node=check_to_variable[edge.check_node],
                check_node=variable_to_check[edge.variable_node],
                pauli_checked=edge.pauli_checked.dual(),
            )
            for edge in graph.edges
        }

        old_to_new_nodes = {**check_to_variable, **variable_to_check}

        return (
            TannerGraph(
                variable_nodes=dual_variable_nodes,
                check_nodes=dual_check_nodes,
                edges=dual_edges,
            ),
            old_to_new_nodes,
        )

    def indexed_dual_graph(
        graph: TannerGraph, index: Dict[int, TannerNode]
    ) -> Tuple[TannerGraph, Dict[int, TannerNode]]:
        """Return the dual Tanner graph and an assignment of integers, corresponding to the structure of the original graph."""
        dual_graph, old_to_new_nodes = TannerGraphAlgebra.dual_graph(graph)
        n_index = {idx: old_to_new_nodes[node] for idx, node in index.items()}
        return dual_graph, n_index

    def index_nodes(graph: TannerGraph) -> Dict[int, TannerNode]:
        """Assigns a unique integer index to each node in the Tanner graph."""
        index = {}
        current_index = 0
        for node in graph.variable_nodes.union(graph.check_nodes):
            index[current_index] = node
            current_index += 1
        return index

    def shortest_path(
        graph: TannerGraph,
        start: TannerNode,
        end: TannerNode,
        check_type: PauliChar | None = None,
    ) -> List[TannerNode]:
        """
        Returns the shortest path between two nodes in the Tanner graph using breadth-first search (BFS).
        The path is returned as a list of nodes, from `start` node to `end` node.
        If no path exists, an empty list is returned.

        Parameters
        ----------
        check_type : PauliChar | None
            When set, only check nodes whose ``check_type`` matches this value
            may appear in the path.  Variable nodes are always allowed.
        """
        if start not in graph.variable_nodes and start not in graph.check_nodes:
            raise ValueError("Start node is not in the Tanner graph.")
        if end not in graph.variable_nodes and end not in graph.check_nodes:
            raise ValueError("End node is not in the Tanner graph.")

        queue = deque([start])
        visited = {start}
        parent = {start: None}

        while queue:
            current = queue.popleft()

            if current == end:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]

            for neighbor in graph.get_neighbourhood(current):
                if neighbor not in visited:
                    if (
                        check_type is not None
                        and isinstance(neighbor, CheckNode)
                        and neighbor.check_type != check_type
                    ):
                        continue
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        return []

    @classmethod
    def best_meeting_node(
        cls,
        graph: TannerGraph,
        subset: List[TannerNode],
        check_type: PauliChar | None = None,
    ) -> Tuple[TannerNode, Dict[TannerNode, List[TannerNode]]]:
        """
        Finds the node minimizing the sum of shortest path distances from all nodes
        in `subset`. Returns the meeting node and the shortest path from each subset
        node to it.

        Parameters
        ----------
        check_type : PauliChar | None
            When set, forwarded to ``shortest_path`` so that only check nodes of
            the given type may be used while routing.

        Returns
        -------
        Tuple[TannerNode, Dict[TannerNode, List[TannerNode]]]
            The meeting node, and a dictionary mapping each node in `subset` to its
            shortest path to the meeting node.
        """

        all_nodes = list(graph.variable_nodes) + list(graph.check_nodes)

        best_node = None
        best_cost = float("inf")
        best_paths = {}

        for candidate in all_nodes:

            total_cost = 0
            paths = {}
            valid = True

            for s in subset:
                path = cls.shortest_path(graph, s, candidate, check_type=check_type)

                if not path:
                    valid = False
                    break

                paths[s] = path
                total_cost += len(path) - 1  # number of edges

            if valid and total_cost < best_cost:
                best_cost = total_cost
                best_node = candidate
                best_paths = paths

        if best_node is None:
            return None, {}

        return best_node, best_paths

    def visualize(graph: TannerGraph, periodic: bool = False):
        """Plot the TannerGraph, with variable nodes as circles and check nodes as squares. Edges are colored according to the Pauli type they check (e.g. X in red, Z in blue, Y in purple). Node tags are displayed next to the nodes for clarity."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import ConnectionPatch

        edge_color = {
            PauliChar.X: "tab:red",
            PauliChar.Z: "tab:blue",
            PauliChar.Y: "tab:purple",
        }

        all_nodes = list(graph.variable_nodes) + list(graph.check_nodes)
        if not all_nodes:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_title("Empty Tanner Graph")
            ax.set_axis_off()
            return fig, ax

        coord_lengths = {len(node.coordinates) for node in all_nodes}
        if 0 in coord_lengths and len(coord_lengths) > 1:
            raise ValueError("Either all nodes must have coordinates or none.")
        if len(coord_lengths) > 1:
            raise ValueError("All node coordinates must have the same dimension.")

        dim = coord_lengths.pop()
        if dim not in {0, 2, 3}:
            raise ValueError("Node coordinates must be empty, 2D, or 3D.")

        def _draw_single_axis(ax, pos):
            for edge in graph.edges:
                x1, y1 = pos[edge.variable_node]
                x2, y2 = pos[edge.check_node]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=edge_color.get(edge.pauli_checked, "gray"),
                    linewidth=1.6,
                    alpha=0.85,
                    zorder=1,
                )

            var_nodes = sorted(graph.variable_nodes, key=lambda n: n.tag)
            check_nodes = sorted(graph.check_nodes, key=lambda n: n.tag)

            ax.scatter(
                [pos[n][0] for n in var_nodes],
                [pos[n][1] for n in var_nodes],
                marker="o",
                s=120,
                color="black",
                zorder=3,
                label="Variable",
            )
            ax.scatter(
                [pos[n][0] for n in check_nodes],
                [pos[n][1] for n in check_nodes],
                marker="s",
                s=130,
                color="dimgray",
                zorder=3,
                label="Check",
            )

            for node in var_nodes + check_nodes:
                x, y = pos[node]
                ax.text(x + 0.03, y + 0.03, node.tag, fontsize=8)

            legend_items = [
                Line2D(
                    [0], [0], marker="o", linestyle="", color="black", label="Variable"
                ),
                Line2D(
                    [0], [0], marker="s", linestyle="", color="dimgray", label="Check"
                ),
                Line2D([0], [0], color="tab:red", label="X edge"),
                Line2D([0], [0], color="tab:blue", label="Z edge"),
                Line2D([0], [0], color="tab:purple", label="Y edge"),
            ]
            ax.legend(handles=legend_items, loc="best", fontsize=8)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.12, linewidth=0.6, linestyle=":")

        if dim == 0:
            fig, ax = plt.subplots(figsize=(9, 5))

            var_nodes = sorted(graph.variable_nodes, key=lambda n: n.tag)
            check_nodes = sorted(graph.check_nodes, key=lambda n: n.tag)
            css_like = all(
                c.check_type in {PauliChar.X, PauliChar.Z} for c in check_nodes
            )

            pos = {}
            for i, node in enumerate(var_nodes):
                pos[node] = (0.0, float(-i))

            if css_like:
                x_checks = [c for c in check_nodes if c.check_type == PauliChar.X]
                z_checks = [c for c in check_nodes if c.check_type == PauliChar.Z]
                for i, node in enumerate(sorted(x_checks, key=lambda n: n.tag)):
                    pos[node] = (-1.0, float(-i))
                for i, node in enumerate(sorted(z_checks, key=lambda n: n.tag)):
                    pos[node] = (1.0, float(-i))
            else:
                for i, node in enumerate(check_nodes):
                    pos[node] = (1.0, float(-i))

            _draw_single_axis(ax, pos)
            ax.set_title("Tanner Graph (Bipartite Layout)")
            return fig, ax

        if dim == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            pos = {
                node: (node.coordinates[0], node.coordinates[1]) for node in all_nodes
            }
            if not periodic:
                _draw_single_axis(ax, pos)
                ax.set_title("Tanner Graph (2D Coordinates)")
                return fig, ax

            # Periodic rendering for toroidal layouts: wrap-across edges are drawn
            # as two boundary-touching segments so periodic connectivity is explicit.
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = (max_x - min_x) + 1
            span_y = (max_y - min_y) + 1
            pad = 0.45

            for edge in graph.edges:
                x1, y1 = pos[edge.variable_node]
                x2, y2 = pos[edge.check_node]
                color = edge_color.get(edge.pauli_checked, "gray")

                dx = x2 - x1
                dy = y2 - y1
                wraps_x = abs(dx) > (span_x / 2)
                wraps_y = abs(dy) > (span_y / 2)

                if not wraps_x and not wraps_y:
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        linewidth=1.6,
                        alpha=0.85,
                        zorder=1,
                    )
                    continue

                if wraps_x and not wraps_y:
                    if dx > 0:
                        ax.plot(
                            [x1, min_x - pad],
                            [y1, y1],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                        ax.plot(
                            [max_x + pad, x2],
                            [y2, y2],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                    else:
                        ax.plot(
                            [x1, max_x + pad],
                            [y1, y1],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                        ax.plot(
                            [min_x - pad, x2],
                            [y2, y2],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                    continue

                if wraps_y and not wraps_x:
                    if dy > 0:
                        ax.plot(
                            [x1, x1],
                            [y1, min_y - pad],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                        ax.plot(
                            [x2, x2],
                            [max_y + pad, y2],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                    else:
                        ax.plot(
                            [x1, x1],
                            [y1, max_y + pad],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                        ax.plot(
                            [x2, x2],
                            [min_y - pad, y2],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                    continue

                # Fallback for rare diagonal wrap cases.
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=color,
                    linewidth=1.2,
                    alpha=0.6,
                    zorder=1,
                    linestyle="--",
                )

            var_nodes = sorted(graph.variable_nodes, key=lambda n: n.tag)
            check_nodes = sorted(graph.check_nodes, key=lambda n: n.tag)

            ax.scatter(
                [pos[n][0] for n in var_nodes],
                [pos[n][1] for n in var_nodes],
                marker="o",
                s=120,
                color="black",
                zorder=3,
                label="Variable",
            )
            ax.scatter(
                [pos[n][0] for n in check_nodes],
                [pos[n][1] for n in check_nodes],
                marker="s",
                s=130,
                color="dimgray",
                zorder=3,
                label="Check",
            )

            for node in var_nodes + check_nodes:
                x, y = pos[node]
                ax.text(x + 0.03, y + 0.03, node.tag, fontsize=8)

            legend_items = [
                Line2D(
                    [0], [0], marker="o", linestyle="", color="black", label="Variable"
                ),
                Line2D(
                    [0], [0], marker="s", linestyle="", color="dimgray", label="Check"
                ),
                Line2D([0], [0], color="tab:red", label="X edge"),
                Line2D([0], [0], color="tab:blue", label="Z edge"),
                Line2D([0], [0], color="tab:purple", label="Y edge"),
            ]
            ax.legend(handles=legend_items, loc="best", fontsize=8)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.12, linewidth=0.6, linestyle=":")
            ax.set_xlim(min_x - (pad + 0.15), max_x + (pad + 0.15))
            ax.set_ylim(min_y - (pad + 0.15), max_y + (pad + 0.15))
            ax.set_title("Tanner Graph (2D Toroidal Coordinates)")
            return fig, ax

        planes = sorted({node.coordinates[2] for node in all_nodes})
        fig, axes = plt.subplots(
            1,
            len(planes),
            figsize=(6 * len(planes), 5),
            squeeze=False,
        )
        axes_list = list(axes[0])
        plane_to_ax = {plane: axes_list[i] for i, plane in enumerate(planes)}

        pos_2d = {
            node: (node.coordinates[0], node.coordinates[1]) for node in all_nodes
        }
        cross_plane_edges = []
        for edge in graph.edges:
            p_var = edge.variable_node.coordinates[2]
            p_chk = edge.check_node.coordinates[2]
            if p_var == p_chk:
                ax = plane_to_ax[p_var]
                x1, y1 = pos_2d[edge.variable_node]
                x2, y2 = pos_2d[edge.check_node]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=edge_color.get(edge.pauli_checked, "gray"),
                    linewidth=1.6,
                    alpha=0.85,
                    zorder=1,
                )
            else:
                cross_plane_edges.append(edge)

        for plane, ax in plane_to_ax.items():
            plane_var = sorted(
                [n for n in graph.variable_nodes if n.coordinates[2] == plane],
                key=lambda n: n.tag,
            )
            plane_check = sorted(
                [n for n in graph.check_nodes if n.coordinates[2] == plane],
                key=lambda n: n.tag,
            )

            if plane_var:
                ax.scatter(
                    [pos_2d[n][0] for n in plane_var],
                    [pos_2d[n][1] for n in plane_var],
                    marker="o",
                    s=120,
                    color="black",
                    zorder=3,
                )
            if plane_check:
                ax.scatter(
                    [pos_2d[n][0] for n in plane_check],
                    [pos_2d[n][1] for n in plane_check],
                    marker="s",
                    s=130,
                    color="dimgray",
                    zorder=3,
                )

            for node in plane_var + plane_check:
                x, y = pos_2d[node]
                ax.text(x + 0.03, y + 0.03, node.tag, fontsize=8)

            ax.set_title(f"Plane {plane}")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.12, linewidth=0.6, linestyle=":")

        for edge in cross_plane_edges:
            x1, y1 = pos_2d[edge.variable_node]
            x2, y2 = pos_2d[edge.check_node]
            a1 = plane_to_ax[edge.variable_node.coordinates[2]]
            a2 = plane_to_ax[edge.check_node.coordinates[2]]
            connector = ConnectionPatch(
                xyA=(x1, y1),
                xyB=(x2, y2),
                coordsA="data",
                coordsB="data",
                axesA=a1,
                axesB=a2,
                color=edge_color.get(edge.pauli_checked, "gray"),
                linestyle="--",
                linewidth=1.1,
                alpha=0.7,
            )
            fig.add_artist(connector)

        fig.suptitle("Tanner Graph (3D Coordinates by Plane)")
        fig.tight_layout()
        return fig, axes_list
