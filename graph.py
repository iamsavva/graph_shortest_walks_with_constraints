import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    MosekSolver,
    MosekSolverDetails,
    SnoptSolver,
    IpoptSolver,
    SolverOptions,
    CommonSolverOption,
    L2NormCost,
    Binding,
)
from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
    Hyperellipsoid,
)
from pydrake.all import ( # pylint: disable=import-error, no-name-in-module
    MakeSemidefiniteRelaxation,
)  
import numbers
import pydot

from pydrake.symbolic import (  # pylint: disable=import-error, no-name-in-module, unused-import
    Polynomial,
    Variable,
    Variables,
    Expression,
)
from pydrake.math import (  # pylint: disable=import-error, no-name-in-module, unused-import
    ge,
    eq,
    le,
)

import plotly.graph_objects as go  # pylint: disable=import-error
from plotly.express.colors import sample_colorscale  # pylint: disable=import-error
import plotly.graph_objs as go  # pylint: disable=import-error
from plotly.subplots import make_subplots  # pylint: disable=import-error

from tqdm import tqdm
import pickle

from collections import deque
from program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions

from util import (  # pylint: disable=import-error, no-name-in-module, unused-import
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
    ChebyshevCenter,
    make_polyhedral_set_for_bezier_curve,
    get_kth_control_point,
    add_set_membership,
)

from gcs_util import get_edge_name, make_quadratic_cost_function_matrices


class DualVertex:
    def __init__(
        self,
        name: str,
        prog: MathematicalProgram,
        options: ProgramOptions,
        vertex_is_target: bool = False,
    ):
        self.name = name
        self.options = options

        self.edges_in = []  # type: T.List[str]
        self.edges_out = []  # type: T.List[str]

        self.J = None
        self.J_solution = None
        self.vertex_is_target = vertex_is_target

        self.define_potential(prog)

    def add_edge_in(self, name: str):
        assert name not in self.edges_in
        self.edges_in.append(name)

    def add_edge_out(self, name: str):
        assert name not in self.edges_out
        self.edges_out.append(name)

    def define_potential(self, prog: MathematicalProgram):
        """
        Defining indeterminates for x and flow-in violation polynomial, if necesary
        """
        self.J = prog.NewContinuousVariables(1, "J" + self.name)[0]
        if self.vertex_is_target:
            prog.AddLinearConstraint(self.J == 0)
        else:
            prog.AddLinearConstraint(self.J >= 0)

    def push_up(self, prog: MathematicalProgram,  flow_in = 1):
        prog.AddCost(-flow_in * self.J)


class DualEdge:
    def __init__(
        self,
        name: str,
        v_left: DualVertex,
        v_right: DualVertex,
        edge_cost: float,
        options: ProgramOptions,
    ):
        # TODO: pass target convex set into constructor
        self.name = name
        self.left = v_left
        self.right = v_right
        self.edge_cost = edge_cost

        self.options = options

        self.positive_edge_penalties = []
        self.negative_edge_penalties = []

    def add_edge_penalty(self, h: Expression, positive: bool):
        if positive:
            self.positive_edge_penalties.append(h)
        else:
            self.negative_edge_penalties.append(h)

    def define_edge_constraint(self, prog: MathematicalProgram):
        """
        define edge appropriate SOS constraints
        """
        expr = (
            self.right.J
            + self.edge_cost
            - self.left.J
            + np.sum(self.positive_edge_penalties)
            - np.sum(self.negative_edge_penalties)
        )
        prog.AddLinearConstraint(expr >= 0)


class PolynomialDualGCS:
    def __init__(
        self,
        options: ProgramOptions,
    ):
        self.vertices = dict()  # type: T.Dict[str, DualVertex]
        self.edges = dict()  # type: T.Dict[str, DualEdge]
        self.prog = MathematicalProgram()  # type: MathematicalProgram
        self.value_function_solution = None  # type: MathematicalProgramResult
        self.options = options
        
        self.positive_penalty_edge_sets = [] # type: T.List[ T.Tuple[T.List[DualEdge], Variable, int] ]
        self.negative_penalty_edge_sets = [] # type: T.List[ T.Tuple[T.List[DualEdge], Variable, int] ]


    def AddVertex(
        self,
        name: str,
        vertex_is_target:bool = False
    ) -> DualVertex:
        """
        Options will default to graph initialized options if not specified
        """
        assert name not in self.vertices
        # add vertex to policy graph
        v = DualVertex(
            name,
            self.prog,
            self.options,
            vertex_is_target,
        )
        self.vertices[name] = v
        return v

    def MaxCostOverVertex(self, vertex: DualVertex, flow_in:float=1.0):
        vertex.push_up(self.prog, flow_in)


    def BuildTheProgram(self):
        INFO("adding edge constraints")
        for edge in self.edges.values():
            edge.define_edge_constraint(self.prog)

    def get_all_n_step_paths(
        self,
        start_lookahead: int,
        start_vertex: DualVertex,
        previous_vertex: DualVertex = None,
    ) -> T.List[T.List[DualVertex]]:
        """
        find every n-step path from the current vertex.
        """

        paths = []  # type: T.List[T.List[DualVertex]]
        vertex_expand_que = deque([(start_vertex, [start_vertex], start_lookahead)])
        while len(vertex_expand_que) > 0:
            vertex, path, lookahead = vertex_expand_que.pop()  # type: DualVertex
            if lookahead == 0:
                paths.append(path)
            else:
                if vertex.vertex_is_target:
                    paths.append(path)
                else:
                    for edge_name in vertex.edges_out:
                        right_vertex = self.edges[edge_name].right
                        vertex_expand_que.append(
                            (right_vertex, path + [right_vertex], lookahead - 1)
                        )
        return paths

    def get_all_n_step_paths_no_revisits(
        self,
        start_lookahead: int,
        start_vertex: DualVertex,
        already_visited=T.List[DualVertex],
    ) -> T.List[T.List[DualVertex]]:
        """
        find every n-step path without revisits
        there isn't actually a way to incorporate that on the policy level. must add as constraint.
        there is a heuristic
        """
        paths = []  # type: T.List[T.List[DualVertex]]
        vertex_expand_que = deque([(start_vertex, [start_vertex], start_lookahead)])
        while len(vertex_expand_que) > 0:
            vertex, path, lookahead = vertex_expand_que.pop()
            # ran out of lookahead -- stop
            if lookahead == 0:
                paths.append(path)
            else:
                if vertex.vertex_is_target:
                    paths.append(path)
                else:
                    for edge_name in vertex.edges_out:
                        right_vertex = self.edges[edge_name].right
                        # don't do revisits
                        if (
                            right_vertex not in path
                            and right_vertex not in already_visited
                        ):
                            vertex_expand_que.append(
                                (right_vertex, path + [right_vertex], lookahead - 1)
                            )
        return paths

    def AddEdge(
        self,
        v_left: DualVertex,
        v_right: DualVertex,
        edge_cost: float,
    ) -> DualEdge:
        """
        Options will default to graph initialized options if not specified
        """
        edge_name = get_edge_name(v_left.name, v_right.name)
        assert edge_name not in self.edges
        e = DualEdge(
            edge_name,
            v_left,
            v_right,
            edge_cost,
            self.options
        )
        self.edges[edge_name] = e
        v_left.add_edge_out(edge_name)
        v_right.add_edge_in(edge_name)
        return e
    
    def flow_over_edges_at_least_this(self, edges:T.List[DualEdge], flow: int = 1):
        # at least this -- negative edge penalty, max positive cost, min negative 
        h = self.prog.NewContinuousVariables(1, "h")[0]
        self.prog.AddLinearConstraint(h >= 0)
        for edge in edges:
            edge.add_edge_penalty(h, False)

        self.prog.AddLinearCost( - flow * h )

        self.negative_penalty_edge_sets.append((edges, h, flow))

    def flow_over_edges_no_more_than_this(self, edges:T.List[DualEdge], flow: int = 1):
        # no more than this -- positive edge penalty, max negative cost, min positive 
        h = self.prog.NewContinuousVariables(1)[0]
        for edge in edges:
            edge.add_edge_penalty(h, True)

        self.prog.AddLinearCost( flow * h )

        self.positive_penalty_edge_sets.append((edges, h, flow))
        
    

    def SolvePolicy(self) -> MathematicalProgramResult:
        """
        Synthesize a policy over the graph.
        Policy is stored in the solution: you'd need to extract it per vertex.
        """
        self.BuildTheProgram()

        timer = timeit()
        mosek_solver = MosekSolver()
        solver_options = SolverOptions()

        # set the solver tolerance gaps
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
            self.options.value_synthesis_MSK_DPAR_INTPNT_CO_TOL_REL_GAP,
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            self.options.value_synthesis_MSK_DPAR_INTPNT_CO_TOL_PFEAS,
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            self.options.value_synthesis_MSK_DPAR_INTPNT_CO_TOL_DFEAS,
        )

        if self.options.value_synthesis_use_robust_mosek_parameters:
            solver_options.SetOption(
                MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3
            )
            solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)

        # solve the program
        self.value_function_solution = mosek_solver.Solve(
            self.prog, solver_options=solver_options
        )
        timer.dt("Solve")
        diditwork(self.value_function_solution)

        for v in self.vertices.values():
            v.J_solution = self.value_function_solution.GetSolution(v.J)
        
        # for (edges, h, flow) in self.positive_penalty_edge_sets:

        return self.value_function_solution