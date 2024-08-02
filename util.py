import typing as T
import numpy.typing as npt

from colorama import Fore
import time
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

from IPython.display import Markdown, display
from scipy.linalg import block_diag


def ERROR(*texts, verbose: bool = True):
    if verbose:
        print(Fore.RED + " ".join([str(text) for text in texts]))


def WARN(*texts, verbose: bool = True):
    if verbose:
        print(Fore.YELLOW + " ".join([str(text) for text in texts]))


def INFO(*texts, verbose: bool = True):
    if verbose:
        print(Fore.BLUE + " ".join([str(text) for text in texts]))


def YAY(*texts, verbose: bool = True):
    if verbose:
        print(Fore.GREEN + " ".join([str(text) for text in texts]))


def latex(prog: MathematicalProgram):
    display(Markdown(prog.ToLatex()))


def diditwork(solution: MathematicalProgramResult, verbose=True):
    if solution.is_success():
        printer = YAY
        printer("solve successful!", verbose=verbose)
    else:
        printer = ERROR
        printer("solve failed", verbose=verbose)
    printer(solution.get_optimal_cost(), verbose=verbose)
    printer(solution.get_solution_result(), verbose=verbose)
    printer("Solver is", solution.get_solver_id().name(), verbose=verbose)
    details = solution.get_solver_details()  # type: MosekSolverDetails
    if isinstance(details, MosekSolverDetails):
        printer(details, verbose=verbose)
        printer("time", details.optimizer_time, verbose=verbose)
        printer("rescode", details.rescode, verbose=verbose)
        printer("solution_status", details.solution_status, verbose=verbose)
    return solution.is_success()


def all_possible_combinations_of_items(item_set: T.List[str], num_items: int):
    """
    Recursively generate a set of all possible ordered strings of items of length num_items.
    """
    if num_items == 0:
        return [""]
    result = []
    possible_n_1 = all_possible_combinations_of_items(item_set, num_items - 1)
    for item in item_set:
        result += [item + x for x in possible_n_1]
    return result


def integrate_a_polynomial_on_a_box(
    poly: Polynomial, x: Variables, lb: npt.NDArray, ub: npt.NDArray
):
    assert len(lb) == len(ub)
    # compute by integrating each monomial term
    monomial_to_coef_map = poly.monomial_to_coefficient_map()
    expectation = Expression(0)
    for monomial in monomial_to_coef_map.keys():
        coef = monomial_to_coef_map[monomial]
        poly = Polynomial(monomial)
        for i in range(len(x)):
            x_min, x_max, x_val = lb[i], ub[i], x[i]
            integral_of_poly = poly.Integrate(x_val)
            poly = integral_of_poly.EvaluatePartial(
                {x_val: x_max}
            ) - integral_of_poly.EvaluatePartial({x_val: x_min})
        expectation += coef * poly.ToExpression()
    if not isinstance(expectation, float):
        ERROR("integral is not a value, it should be")
        ERROR(expectation)
        return None
    return expectation


class timeit:
    def __init__(self):
        self.times = []
        self.times.append(time.time())
        self.totals = 0
        self.a_start = None

    def dt(self, descriptor=None, print_stuff=True):
        self.times.append(time.time())
        if print_stuff:
            if descriptor is None:
                INFO("%.3fs since last time-check" % (self.times[-1] - self.times[-2]))
            else:
                descriptor = str(descriptor)
                INFO(descriptor + " took %.3fs" % (self.times[-1] - self.times[-2]))
        return self.times[-1] - self.times[-2]

    def T(self, descriptor=None):
        self.times.append(time.time())
        if descriptor is None:
            INFO("%.3fs since the start" % (self.times[-1] - self.times[0]))
        else:
            INFO(
                descriptor
                + " took %.3fs since the start" % (self.times[-1] - self.times[0])
            )

    def start(self):
        self.a_start = time.time()

    def end(self):
        self.totals += time.time() - self.a_start
        self.a_start = None

    def total(self, descriptor=None):
        INFO("All " + descriptor + " took %.3fs" % (self.totals))


def ChebyshevCenter(
    poly: HPolyhedron, center: npt.NDArray = None
) -> T.Tuple[bool, npt.NDArray, float]:

    # Ax <= b
    m = poly.A().shape[0]
    n = poly.A().shape[1]

    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(n, "x")
    r = prog.NewContinuousVariables(1, "r")
    prog.AddLinearCost(np.array([-1]), 0, r)

    big_num = 100000

    prog.AddBoundingBoxConstraint(0, big_num, r)

    if center is not None:
        assert len(center) == n
        prog.AddLinearConstraint(eq(x, center))

    a = np.zeros((1, n + 1))
    for i in range(m):
        a[0, 0] = np.linalg.norm(poly.A()[i, :])
        a[0, 1:] = poly.A()[i, :]
        prog.AddLinearConstraint(
            a, -np.array([big_num]), np.array([poly.b()[i]]), np.append(r, x)
        )

    result = Solve(prog)
    if not result.is_success():
        return False, None, None
    else:
        return True, result.GetSolution(x), result.GetSolution(r)[0]


def offset_hpoly_inwards(hpoly: HPolyhedron, eps: float = 1e-5) -> HPolyhedron:
    A, b = hpoly.A(), hpoly.b()
    return HPolyhedron(A, b - eps)


def have_full_dimensional_intersection(
    hpoly1: HPolyhedron, hpoly2: HPolyhedron
) -> bool:
    intersection = hpoly1.Intersection(hpoly2)
    inward_intersection = offset_hpoly_inwards(intersection)
    return not inward_intersection.IsEmpty()


def make_polyhedral_set_for_bezier_curve(
    hpoly: HPolyhedron, num_control_points: int
) -> HPolyhedron:
    A, b = hpoly.A(), hpoly.b()
    m, n = A.shape
    k = num_control_points
    bigA = np.zeros((k * m, k * n))
    bigb = np.zeros(k * m)
    for i in range(k):
        bigA[m * i : m * (i + 1), n * i : n * (i + 1)] = A
        bigb[m * i : m * (i + 1)] = b
    return HPolyhedron(bigA, bigb)


def get_kth_control_point(
    bezier: npt.NDArray, k: int, num_control_points: int
) -> npt.NDArray:
    state_dim = len(bezier) // num_control_points
    return bezier[k * state_dim : (k + 1) * state_dim]


def add_set_membership(
    prog: MathematicalProgram,
    convex_set: ConvexSet,
    x: npt.NDArray,
    ellipsoid_as_lorentz=True,
) -> None:
    if isinstance(convex_set, HPolyhedron):
        # prog.AddLinearConstraint(le( convex_set.A().dot(x), convex_set.b()))
        prog.AddLinearConstraint(
            convex_set.A(), -np.infty * np.ones(len(convex_set.b())), convex_set.b(), x
        )
    elif isinstance(convex_set, Hyperrectangle):
        hpoly = convex_set.MakeHPolyhedron()
        # prog.AddLinearConstraint(le( hpoly.A().dot(x), hpoly.b()))
        prog.AddLinearConstraint(
            hpoly.A(), -np.infty * np.ones(len(hpoly.b())), hpoly.b(), x
        )
    elif isinstance(convex_set, Hyperellipsoid):
        A, c = convex_set.A(), convex_set.center()
        if ellipsoid_as_lorentz:
            lhs = convex_set.A().dot(x) - convex_set.A().dot(convex_set.center())
            rhs = [1]
            prog.AddLorentzConeConstraint(np.hstack((rhs, lhs)))
        else:
            prog.AddQuadraticConstraint((x - c).dot(A.T @ A).dot(x - c), -np.inf, 1)
    elif isinstance(convex_set, Point):
        # prog.AddLinearConstraint(eq(x, convex_set.x()))
        # prog.AddLinearEqualityConstraint(np.eye(len(x)),convex_set.x(), x)
        prog.AddLinearEqualityConstraint(x, convex_set.x())
    else:
        assert False, "bad set in add_set_membership"


def concatenate_polyhedra(sets: T.List[ConvexSet]) -> HPolyhedron:
    res = None
    for cset in sets:
        assert isinstance(cset, Hyperrectangle) or isinstance(cset, HPolyhedron)
        if isinstance(cset, Hyperrectangle):
            hpoly = cset.MakeHPolyhedron()
        else:
            hpoly = cset

        if res is None:
            res = hpoly
        else:
            new_A = block_diag(res.A(), hpoly.A())
            new_b = np.vstack(
                (
                    res.b().reshape((len(res.b()), 1)),
                    hpoly.b().reshape((len(hpoly.b()), 1)),
                )
            )
            res = HPolyhedron(new_A, new_b)
    return res


def recenter_convex_set(convex_set: ConvexSet, center: npt.NDArray) -> ConvexSet:
    if isinstance(convex_set, HPolyhedron):
        assert False, "can't handle polyhedra:("
    elif isinstance(convex_set, Hyperrectangle):
        half_width = convex_set.Center() - convex_set.lb()
        return Hyperrectangle(center - half_width, center + half_width)
    elif isinstance(convex_set, Hyperellipsoid):
        return Hyperellipsoid(convex_set.A(), center)
    elif isinstance(convex_set, Point):
        return Point(center)
    else:
        assert False, "bad set in add_set_membership"
