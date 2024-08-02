import typing as T

import numpy as np
import numpy.typing as npt  # pylint: disable=unused-import
import pydot  # pylint: disable=import-error

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module
    MathematicalProgramResult,
)

from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    GraphOfConvexSets,
)

from util import (
    timeit,
    INFO,
    YAY,
    ERROR,
    WARN,
    diditwork,
)  # pylint: disable=unused-import

from pydrake.symbolic import (  # pylint: disable=import-error, no-name-in-module, unused-import
    Expression,
)


def get_vertex_name(t: int, set_name: str) -> str:
    return "T" + str(t) + " " + set_name


def get_mode_from_name(vertex_name: str) -> str:
    return vertex_name.split(" ")[1]


def get_edge_name(left_name: str, right_name: str) -> str:
    return left_name + " " + right_name


def make_quadratic_cost_function(x_star: npt.NDArray, Q: npt.NDArray) -> T.Callable:
    assert Q.shape[0] == Q.shape[1] == len(x_star)

    def cost_function(x):
        assert len(x) == len(x_star)
        return 0.5 * (x - x_star).dot(Q).dot(x - x_star)

    return cost_function


def make_quadratic_cost_function_matrices(
    x_star: npt.NDArray, Q: npt.NDArray
) -> T.Tuple[float, npt.NDArray, npt.NDArray]:
    c2 = 0.5 * Q
    c1 = -x_star.dot(Q)
    c0 = 0.5 * x_star.dot(Q).dot(x_star)
    return c0, c1, c2


def make_quadratic_state_control_cost_function(
    x_star: npt.NDArray, u_star: npt.NDArray, Q: npt.NDArray, R: npt.NDArray
) -> T.Callable:
    state_cost = make_quadratic_cost_function(x_star, Q)
    control_cost = make_quadratic_cost_function(u_star, R)

    def cost_function(x, u):
        assert len(x) == len(x_star)
        assert len(u) == len(u_star)
        return state_cost(x) + control_cost(u)

    return cost_function


def find_path_to_target(
    solution: MathematicalProgramResult,
    edges: T.List[GraphOfConvexSets.Edge],
    start: GraphOfConvexSets.Vertex,
    target: GraphOfConvexSets.Vertex,
) -> T.Tuple[T.List[GraphOfConvexSets.Vertex], T.List[GraphOfConvexSets.Edge]]:
    """Given a set of active edges, find a path from start to target.
    Returns:
        T.Tuple[T.List[GraphOfConvexSets.Vertex], T.List[GraphOfConvexSets.Edge]]: vertex path, edge path
    """
    # seed
    np.random.seed(1)
    # get edges out of the start vertex
    edges_out = [e for e in edges if e.u() == start]
    # get flows out of start vertex
    flows_out = np.array([solution.GetSolution(e.phi()) for e in edges_out])
    proabilities = np.where(flows_out < 0, 0, flows_out)  # fix numerical errors
    proabilities /= sum(proabilities)  # normalize
    # pick next edge at random
    current_edge = np.random.choice(edges_out, 1, p=proabilities)[0]
    # get the next vertex and continue
    v = current_edge.v()
    # check to see if target has been reached
    target_reached = v == target
    # return the list of vertices and edges along the path
    if target_reached:
        return [start] + [v], [current_edge]
    else:
        v, e = find_path_to_target(solution, edges, v, target)
        return [start] + v, [current_edge] + e


def get_random_solution_path(
    gcs: GraphOfConvexSets,
    solution: MathematicalProgramResult,
    start: GraphOfConvexSets.Vertex,
    target: GraphOfConvexSets.Vertex,
) -> T.Tuple[T.List[GraphOfConvexSets.Vertex], T.List[GraphOfConvexSets.Edge]]:
    """Extract a path from a solution to a gcs program."""
    flow_variables = [e.phi() for e in gcs.Edges()]
    flow_results = [solution.GetSolution(p) for p in flow_variables]
    active_edges = [edge for edge, flow in zip(gcs.Edges(), flow_results) if flow > 0.0]
    return find_path_to_target(solution, active_edges, start, target)


def get_solution_values(
    gcs: GraphOfConvexSets,
    solution: MathematicalProgramResult,
    start: GraphOfConvexSets.Vertex,
    target: GraphOfConvexSets.Vertex,
):
    vertex_trajectory, _ = get_random_solution_path(gcs, solution, start, target)
    value_trajectory = [solution.GetSolution(v.x()) for v in vertex_trajectory]
    return value_trajectory


def solution_not_tight(gcs, solution):
    flow_variables = [e.phi() for e in gcs.Edges()]
    flow_results = [solution.GetSolution(p) for p in flow_variables]
    not_tight = np.any(
        np.logical_and(0.02 < np.array(flow_results), np.array(flow_results) < 0.98)
    )
    not_tight_edges = np.sum(
        np.logical_and(0.02 < np.array(flow_results), np.array(flow_results) < 0.98)
    )
    total_edges = len(gcs.Edges())
    if not_tight:
        WARN(not_tight_edges, "not tight edges out of", total_edges, "total.")
        non_zero_edges = np.sum(0.02 < np.array(flow_results))
        WARN(non_zero_edges, "non-zero edges out of", total_edges, "total.")
    return not_tight


def plot_a_gcs(gcs, solution=None, graph_name="temp"):
    if solution is None:
        graphviz = gcs.GetGraphvizString()
    else:
        graphviz = gcs.GetGraphvizString(solution, True, precision=2)

    data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
    data.write_png(graph_name + ".png")
    data.write_svg(graph_name + ".svg")
