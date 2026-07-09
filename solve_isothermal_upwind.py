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
    locate_dofs_geometrical,
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
from dolfinx import mesh
from petsc4py import PETSc
import pickle

def solve_Richards(h_w, h_w_old, snes, problem, b, J, delta_t, t, min_dt=1e-2):
    repeat_time_step = False
    h_w.x.array[:] = h_w_old.x.array
    snes.setFunction(problem.F, b)  # assemble residual
    snes.setJacobian(problem.J, J)  # assemble Jacobian

    # Set options
    snes.setType("newtonls")
    snes.getLineSearch().setType(PETSc.SNESLineSearch.Type.BT)
    snes.setTolerances(rtol=1e-4, atol=1e-11, max_it=50)
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

    converged = snes.getConvergedReason()
    num_iter = snes.getIterationNumber()
    if converged <= 0:
        # Newton actually failed (not just "slow") -> always halve dt,
        # no floor check here.
        if float(delta_t.value) <= min_dt:
            raise RuntimeError(
                f"Solver failed to converge (reason {converged}) even at "
                f"the minimum time step {min_dt} s, t={t/3600:.2f} h."
            )
        delta_t.value = max(0.5*float(delta_t.value), min_dt)
        print(f"Newton diverged (reason {converged}), halving dt to "
              f"{delta_t.value:.4f} s and retrying t={t/3600:.2f} h.")
        return h_w, True, delta_t
    
    sol_vec.copy(h_w.x.petsc_vec)  # copy solution into h_w
    h_w.x.scatter_forward()

    # Converged, but check if it was "slow" and should shrink dt anyway
    if num_iter > 10 and float(delta_t.value) > min_dt:
        delta_t.value = max(0.5*float(delta_t.value), min_dt)
        return h_w, True, delta_t

    if num_iter < 3 and float(delta_t.value) < 3600:
        delta_t.value = min(float(delta_t.value)*1.2, 3600)

    print(
        f"Solver converged after {num_iter} iterations, reason {converged}. "
        f"dt = {delta_t.value:.2f} s at t={t/3600:.2f} h."
    )
    return h_w, False, delta_t

def solve_system(
        geom, delta_x, boundaries, bc_dict,
        layer_params=None, delta_t=7, T_end=24*60*60,
        save_tmp = False, filename=None):
    nx = int(6/delta_x)
    nz = int(3/delta_x)
    print(f"Resolution is dx = dz = {delta_x} m, giving nx = {nx}, nz = {nz}")
    domain = geom.make_domain(nx, nz)
    V_hw = functionspace(domain, ("CG", 1))
    Q = functionspace(domain, ("DG", 0))
    v_hw = TestFunction(V_hw)
    x = SpatialCoordinate(domain)
    # Get parameters
    p = Parameter(domain, layer_params)

    # Set up time iteration
    delta_t = Constant(domain, PETSc.ScalarType(delta_t))
    t = 0.0

    # Initial conditions
    h_w_ini = -0.3
    h_w_old = Function(V_hw)
    h_w_old.name = "h_w_old"
    h_w_old.x.array[:] = h_w_ini*np.ones_like(h_w_old.x.array)
    phi = Function(Q)
    phi.name = "phi"
    phi.x.array[:] = 0.468*np.ones_like(phi.x.array)
    krel = Function(Q)
    krel.name = "krel"

    # Trial function
    h_w = Function(V_hw)

    # Weak formulation
    F_hw = (
        v_hw * (p.theta(p.S_e(h_w), phi) -
                p.theta(p.S_e(h_w_old), phi)) / delta_t * dx
        + dot(grad(v_hw), (p.K_s(phi)*krel*grad(x[1]+h_w))) * dx
    )
    bc = BoundaryCondition(domain, boundaries)
    for d in bc_dict.values():
        if d["variable"] == "h_w":
            d["functionspace"] = V_hw
            d["testfunction"] = v_hw
            d["problem"] = F_hw
    bcs = bc.make_boundary_condition(bc_dict)
    F_hw += bcs["top"]

    # Create Newton solver
    snes = PETSc.SNES().create()
    # Set up nonlinear problem
    problem_hw = NonlinearPDE_SNESProblem(F_hw, h_w, bc=[])
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
        "krel": [],
    }
    tmp["h_w"].append(h_w_old.x.array.copy())
    tmp["times"].append(t)
    tmp["krel"].append(krel.x.array.copy())
    next_saving_time = 3600

    # Time loop
    while t <= T_end:
        # upwind krel
        new_krel = p.upwind_krel(h_w_old, domain)
        krel.x.array[:] = new_krel.x.array
        krel.x.scatter_forward()
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
            tmp["krel"].append(krel.x.array.copy())

        h_w_old.x.array[:] = h_w.x.array
        h_w_old.x.scatter_forward()
        t += float(delta_t.value)

    if save_tmp:
        # save final data
        tmp["h_w"].append(h_w.x.array.copy())
        tmp["times"].append(t)
        tmp["krel"].append(krel.x.array.copy())
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
def sides(x): 
    return np.logical_or(np.isclose(x[0], P0[0]), np.isclose(x[0], P1[0]))

def top(x):
    return np.isclose(x[1], slope*x[0]+P3[1])

boundaries = {
    1: on_dirichlet,
    2: top,
}
bc_dict = {
    "top": {
        "marker": 2, "name": "Neumann", "value": -2e-9, "variable": "h_w"},
}

layer_params = {
    1: {
        "d_i": 2.31e-4,
        "rho_s": 387,
        "locator": lambda x: x[1] >= slope*x[0] + P3[1]/2 - 1e-14},
    2: {
        "d_i": 4.21e-4,
        "rho_s": 489,
        "locator": lambda x: x[1] < slope*x[0] + P3[1]/2 - 1e-14}
}
filename = "./Masterarbeit/solutions/isothermal_test_newparams.pkl"
solve_system(geom, delta_x, boundaries, bc_dict, save_tmp=True, filename=filename, layer_params=layer_params)



