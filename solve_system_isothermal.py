import numpy as np
from parametrizations import Parameter
from boundary_condition import BoundaryCondition
from geometry_class import Geometry
from nonlinear_snes_problem import NonlinearPDE_SNESProblem
from dolfinx.fem import (
    functionspace,
    Function,
    Constant,
    form,
)
from dolfinx.fem.petsc import (
    create_matrix, create_vector,
    assemble_matrix, assemble_vector,
    apply_lifting, set_bc,
    LinearProblem,
)
from ufl import (
    grad, dx, dot,
    SpatialCoordinate, TestFunction, TrialFunction,
    rhs, lhs, system,
)
from petsc4py import PETSc
import pickle


def solve_Richards(h_w, h_w_old, snes, problem, b, J, delta_t, t):
    repeat_time_step = False
    h_w.x.array[:] = h_w_old.x.array
    snes.setFunction(problem.F, b)  # assemble residual
    snes.setJacobian(problem.J, J)  # assemble Jacobian

    # Set options
    snes.setType("newtonls")
    snes.getLineSearch().setType(PETSc.SNESLineSearch.Type.BT)
    snes.setTolerances(rtol=1e-4, atol=1e-11, max_it=20)
    ksp = snes.getKSP()
    ksp.setType("gmres")  # iterative solver
    ksp.setTolerances(rtol=1e-4)
    ksp.setErrorIfNotConverged(True)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.getPC().setHYPREType("boomeramg")

    sol_vec = h_w.x.petsc_vec.copy()  # create solution vector
    sol_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)
    snes.solve(None, sol_vec)  # solve, store solution in solution vector

    sol_vec.copy(h_w.x.petsc_vec)  # copy solution into h_w
    h_w.x.scatter_forward()

    converged = snes.getConvergedReason()
    num_iter = snes.getIterationNumber()

    # adaptive time stepping:
    if num_iter > 10 and float(delta_t.value) > 1e-1:
        delta_t.value = max(0.5*float(delta_t.value), 1e-1)
        repeat_time_step = True
        return h_w, repeat_time_step, delta_t

    if num_iter < 3 and float(delta_t.value) < 3600:
        delta_t.value = min(float(delta_t.value)*1.2, 3600)
    assert converged > 0, f"Solver did not converge, got {converged}."
    print(
        f"Solver converged after {num_iter} iterations with converged reason {converged}. Time step is {delta_t.value:.2f} s at t={t/3600:.2f} hours."
    )

    return h_w, repeat_time_step, delta_t


def solve_system(
        geom, delta_x, boundaries, bc_dict,
        layer_params=None, delta_t=7, T_end=24*60*60,
        save_tmp = False, filename=None):
    nx = int(6/delta_x)
    nz = int(3/delta_x)
    print(f"Resolution is dx = dz = {delta_x} m, giving nx = {nx}, nz = {nz}")
    domain = geom.make_domain(nx, nz)
    V_hw = functionspace(domain, ("CG", 1))
    v_hw = TestFunction(V_hw)
    x = SpatialCoordinate(domain)
    # Get parameters
    p = Parameter(domain, layer_params)
    bc = BoundaryCondition(domain, boundaries)
    for d in bc_dict.values():
        if d["variable"] == "h_w":
            d["functionspace"] = V_hw
            d["testfunction"] = v_hw
    bcs = bc.make_boundary_condition(bc_dict)

    # Set up time iteration
    delta_t = Constant(domain, PETSc.ScalarType(delta_t))
    t = 0.0

    # Initial conditions
    h_w_old = Function(V_hw)
    h_w_old.name = "h_w_old"
    h_w_old.x.array[:] = -0.3*np.ones_like(h_w_old.x.array)
    phi = Function(V_hw)
    phi.name = "phi"
    phi.x.array[:] = 0.468*np.ones_like(phi.x.array)

    # Trial function
    h_w = Function(V_hw)

    # Weak formulation
    F_hw = (
        v_hw * (p.theta(p.S_e(h_w), phi) -
                p.theta(p.S_e(h_w_old), phi)) / delta_t * dx
        + dot(grad(v_hw), (p.K_s(phi)*p.k_rel(p.S_e(h_w))*grad(x[1]+h_w)))*dx
        + bcs["top"]
    )

    # Create Newton solver
    snes = PETSc.SNES().create()
    # Set up nonlinear problem
    problem_hw = NonlinearPDE_SNESProblem(F_hw, h_w, bc=bcs["side"])
    b_hw = create_vector(V_hw)
    J_hw = create_matrix(problem_hw.a)

    # Create structure for saving intermediate results
    tmp = {
        "geometry": geom.make_into_dict(),
        "T_end": T_end,
        "parameter": p.make_into_dict(),
        "h_w": [],
        "phi": phi.x.array.copy(),
        "times": [],
    }
    tmp["h_w"].append(h_w_old.x.array.copy())
    tmp["times"].append(t)
    next_saving_time = 3600

    # Time loop
    while t <= T_end:
        # Solve Richards
        h_w, repeat_time_step, delta_t = solve_Richards(
            h_w, h_w_old, snes, problem_hw, b_hw, J_hw, delta_t, t)
        if repeat_time_step:
            continue
        # save temporary data
        if save_tmp and t >= next_saving_time:
            next_saving_time += 3600
            tmp["h_w"].append(h_w.x.array.copy())
            tmp["times"].append(t)

        h_w_old.x.array[:] = h_w.x.array
        t += float(delta_t.value)

    # Destroy PETSc objects
    snes.destroy()
    b_hw.destroy()
    J_hw.destroy()

    if save_tmp:
        # save final data
        tmp["h_w"].append(h_w.x.array.copy())
        tmp["times"].append(t)
        # dump temporary data into pickle file
        with open(filename, "wb") as f:
            pickle.dump(tmp, f)

# Set up geometry
height = 3
length = 6
slope = -1/6
delta_x = 0.1
geom = Geometry(height, length, slope=slope)
[P0, P1, P2, P3] = geom.corner_points
# Set up boundary conditions
def on_dirichlet(x):
    return np.logical_and(np.isclose(x[0], P1[0]), x[1] <= 0)

def top(x):
    return np.isclose(x[1], slope*x[0]+P3[1])

boundaries = {
    1: on_dirichlet,
    2: top,
}
bc_dict = {
    "top": {
        "marker": 2, "name": "Neumann", "value": -2e-9, "variable": "h_w"},
    "side": {
        "marker": 1, "name": "Dirichlet", "value": lambda x: -x[1], 
        "variable": "h_w"}
}

filename = "./solutions/pls_work.pkl"
solve_system(geom, delta_x, boundaries, bc_dict, save_tmp=True, filename=filename)



