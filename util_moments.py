import typing as T
import numpy.typing as npt

import numpy as np

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    SolverOptions,
    CommonSolverOption,
    IpoptSolver,
    SnoptSolver,
    MosekSolver,
    MosekSolverDetails,
    GurobiSolver,
)

from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    HPolyhedron,
    Hyperellipsoid,
    Hyperrectangle,
    ConvexSet,
    Point,
)

from pydrake.math import (  # pylint: disable=import-error, no-name-in-module, unused-import
    ge,
    eq,
    le,
)

from pydrake.all import (
    MakeSemidefiniteRelaxation,
)  # pylint: disable=import-error, no-name-in-module

from util import add_set_membership, latex, diditwork


def make_moment_matrix(m0, m1: npt.NDArray, m2: npt.NDArray):
    assert m2.shape == (len(m1), len(m1))
    assert m1.shape == (len(m1),)
    return np.vstack((np.hstack((m0, m1)), np.hstack((m1.reshape((len(m1), 1)), m2))))


def get_moments_from_matrix(M: npt.NDArray):
    return M[0, 0], M[0, 1:], M[1:, 1:]


def extract_moments_from_vector_of_spectrahedron_prog_variables(
    vector: npt.NDArray, dim: int
) -> T.Tuple[float, npt.NDArray, npt.NDArray]:
    # spectrahedron program variables stores vectors in the following array: m1, m2[triu], m0
    # indecis
    ind1 = dim
    ind2 = ind1 + int(dim * (dim + 1) / 2)
    # get moments
    m1 = vector[:ind1]
    m2 = np.zeros((dim, dim), dtype=vector.dtype)
    triu_ind = np.triu_indices(dim)
    m2[triu_ind[0], triu_ind[1]] = vector[ind1:ind2]
    m2[triu_ind[1], triu_ind[0]] = vector[ind1:ind2]
    m0 = vector[ind2]
    return m0, m1, m2


def get_moment_matrix_for_a_measure_over_set(convex_set: ConvexSet):
    if isinstance(convex_set, Point):
        x = convex_set.x()
        return make_moment_matrix(1, x, np.outer(x, x))
    if isinstance(convex_set, HPolyhedron) or isinstance(convex_set, Hyperrectangle):
        hpoly = (
            convex_set
            if isinstance(convex_set, HPolyhedron)
            else convex_set.MakeHPolyhedron()
        )
        ellipsoid = hpoly.MaximumVolumeInscribedEllipsoid()
    elif isinstance(convex_set, Hyperellipsoid):
        ellipsoid = convex_set
    else:
        assert False, "bad set in get_moment_matrix_for_a_measure_over_set"
    mu = ellipsoid.center()
    sigma = np.linalg.inv(ellipsoid.A().T.dot(ellipsoid.A()))
    # print(np.linalg.eigvals(sigma))
    return make_moment_matrix(1, mu, sigma + np.outer(mu, mu))


def make_product_of_indepent_moment_matrices(
    moment_mat1: npt.NDArray, moment_mat2: npt.NDArray
):
    m0_s, m1_s, m2_s = get_moments_from_matrix(moment_mat1)
    m0_t, m1_t, m2_t = get_moments_from_matrix(moment_mat2)
    assert m0_s == m0_t
    row1 = np.hstack((1, m1_s, m1_t))
    row2 = np.hstack((m1_s.reshape((len(m1_s), 1)), m2_s, np.outer(m1_s, m1_t)))
    row3 = np.hstack((m1_t.reshape((len(m1_t), 1)), np.outer(m1_t, m1_s), m2_t))
    moment_matrix = np.vstack((row1, row2, row3))
    return moment_matrix


def verify_necessary_conditions_for_moments_supported_on_set(
    moment_matrix: npt.NDArray, convex_set: ConvexSet
):
    # check SDP-relaxation feasibility of moment_matrix
    state_dim = convex_set.ambient_dimension()
    assert moment_matrix.shape == (state_dim + 1, state_dim + 1)
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(state_dim)
    add_set_membership(prog, convex_set, x, False)
    sdp_prog = MakeSemidefiniteRelaxation(prog)
    m0, m1, m2 = extract_moments_from_vector_of_spectrahedron_prog_variables(
        sdp_prog.decision_variables(), state_dim
    )
    M = make_moment_matrix(m0, m1, m2)
    sdp_prog.AddLinearConstraint(eq(M, moment_matrix))

    solver = MosekSolver()
    solver_options = SolverOptions()

    solver_options.SetOption(
        MosekSolver.id(),
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
        1e-6,
    )
    solver_options.SetOption(
        MosekSolver.id(),
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
        1e-6,
    )
    solver_options.SetOption(
        MosekSolver.id(),
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
        1e-6,
    )
    solver_options.SetOption(
        MosekSolver.id(),
        "MSK_DPAR_INTPNT_TOL_INFEAS",
        1e-6,
    )

    solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
    solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)

    solution = solver.Solve(sdp_prog, solver_options=solver_options)
    # print(moment_matrix)
    # latex(sdp_prog)
    # diditwork(solution)
    return solution.is_success()
