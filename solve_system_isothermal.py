import numpy as np
from parametrizations_adapted_reswater import Parameter
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
    rhs, lhs, system, conditional, ge,
)
from dolfinx import mesh
from petsc4py import PETSc
import pickle


def solve_Richards(h_w, h_w_old, snes, problem, b, J, delta_t, t):
    min_dt = 1e-3
    max_dt = 2
    repeat_time_step = False
    h_w.x.array[:] = h_w_old.x.array
    snes.setFunction(problem.F, b)  # assemble residual
    snes.setJacobian(problem.J, J)  # assemble Jacobian

    # Set options
    snes.setType("newtonls")
    snes.getLineSearch().setType(PETSc.SNESLineSearch.Type.BT)
    snes.setTolerances(rtol=1e-5, atol=1e-11, max_it=50)
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
    if num_iter > 10 and float(delta_t.value) > min_dt:
        delta_t.value = max(0.5*float(delta_t.value), min_dt)
        repeat_time_step = True
        return h_w, repeat_time_step, delta_t

    if num_iter < 3 and float(delta_t.value) < max_dt:
        delta_t.value = min(float(delta_t.value)*1.2, max_dt)
    assert converged > 0, f"Solver did not converge, got {converged}."
    print(
        f"Solver converged after {num_iter} iterations with converged reason {converged}. Time step is {delta_t.value:.2f} s at t={t/3600:.2f} hours."
    )

    return h_w, repeat_time_step, delta_t

def apply_initial_condition(f, ini, DG0_space=None):
    if callable(ini) and DG0_space is None:
        f.interpolate(ini)
    elif callable(ini) and DG0_space is not None:
        helper = Function(DG0_space)
        helper.interpolate(ini)
        f.interpolate(helper)
    else:
        f.x.array[:] = ini*np.ones_like(f.x.array)
    return f

def solve_system(
        geom, delta_x, boundaries, bc_dict, initial_conditions,
        layer_params=None, delta_t=7, T_end=24*60*60, saving_interval=60*60,
        save_tmp = False, filename=None):
    nx = int(geom.length/delta_x)
    nz = int(geom.height/delta_x)
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
    h_w_old = Function(V_hw)
    h_w_old.name = "h_w_old"
    phi = Function(Q)
    phi.name = "phi"
    for key, ini in initial_conditions.items():
        if key == "h_w":
            h_w_old = apply_initial_condition(h_w_old, ini, Q)
        elif key == "phi":
            phi = apply_initial_condition(phi, ini, Q)

    # Trial function
    h_w = Function(V_hw)

    # DG0 function needed for evaluation
    krel = Function(Q)
    krel.name = "krel"

    # Weak formulation
    F_hw = (
        v_hw * (p.theta(p.S_e(h_w), phi) -
                p.theta(p.S_e(h_w_old), phi)) / delta_t * dx
        + dot(grad(v_hw), (p.K_s(phi)*krel*grad(x[1]+h_w))) * dx
    )
    # Boundary conditions
    print_bc = str(bc_dict)
    bc = BoundaryCondition(domain, boundaries)
    for d in bc_dict.values():
        if d["variable"] == "h_w":
            d["functionspace"] = V_hw
            d["testfunction"] = v_hw
            d["problem"] = F_hw
    bcs = bc.make_boundary_condition(bc_dict)
    bc_D_hw = []
    for key, d in bc_dict.items():
        if d["name"] == "Neumann":
            # update weak formulations with Neumann bc
            if d["variable"] == "h_w":
                F_hw += bcs[key]
        elif d["name"] == "Dirichlet":
            # sort Dirichlet bc after variable
            if d["variable"] == "h_w":
                bc_D_hw.append(bcs[key])
        elif d["name"] == "seepage face":
            # update weak formulations with seepage contribution
            if d["variable"] == "h_w":
                F_hw += (v_hw * p.K_s(phi) / bcs[key]
                          *conditional(ge(h_w, 0), h_w, 0) * bc.ds(d["marker"]))

    # Create Newton solver
    snes = PETSc.SNES().create()
    # Set up nonlinear problem
    problem_hw = NonlinearPDE_SNESProblem(F_hw, h_w, bc=bc_D_hw)
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
        "saving_interval": saving_interval,
        "boundary_condition": print_bc,
        "initial_condition": str(initial_conditions),
    }
    tmp["h_w"].append(h_w_old.x.array.copy())
    tmp["times"].append(t)
    next_saving_time = saving_interval

    # Time loop
    while t <= T_end:
        # Upwind krel
        new_krel = p.upwind_krel(h_w_old, domain)
        krel.x.array[:] = new_krel.x.array.copy()
        krel.x.scatter_forward()
        # Solve Richards
        h_w, repeat_time_step, delta_t = solve_Richards(
            h_w, h_w_old, snes, problem_hw, b_hw, J_hw, delta_t, t)
        if repeat_time_step:
            continue
        #sol_vec.copy(h_w.x.petsc_vec)  # copy solution into h_w1
        h_w.x.scatter_forward()
        # save temporary data
        if save_tmp and t >= next_saving_time:
            next_saving_time += saving_interval
            tmp["h_w"].append(h_w.x.array.copy())
            tmp["times"].append(t)

        h_w_old.x.array[:] = h_w.x.array.copy()
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
        with open("./Masterarbeit/solutions/" + filename + ".pkl", "wb") as f:
            pickle.dump(tmp, f)

# Set up geometry
delta_x = 0.005
height = 0.2
length = 0.05
slope = 0.0

geom = Geometry(height, length, slope=slope)
[P0, P1, P2, P3] = geom.corner_points
# Set up boundary conditions
# def on_dirichlet(x):
#     return np.logical_and(np.isclose(x[0], P1[0]), x[1] <= 0)
# def sides(x): 
#     return np.logical_or(np.isclose(x[0], P0[0]), np.isclose(x[0], P1[0]))

def top(x):
    return np.isclose(x[1], height)
def bottom(x):
    return np.isclose(x[1], 0)

boundaries = {
    1: bottom,
    2: top,
}
bc_dict = {
    "top": {
        "marker": 2, "name": "Neumann", "value": -3.3e-6, "variable": "h_w"},
}

# layer_params = {
#     1: {
#         "d_i": 2.31e-4,
#         "rho_s": 387,
#         "locator": lambda x: x[1] >= slope*x[0] + P3[1]/2 - 1e-14},
#     2: {
#         "d_i": 4.21e-4,
#         "rho_s": 489,
#         "locator": lambda x: x[1] < slope*x[0] + P3[1]/2 - 1e-14}
# }

# layer_params = {
#     "all": {"d_i": 1.5e-3, "rho_s": 501,
#         "locator": lambda x: x[1] == x[1]},
#         }
layer_params = {
    "top": {"d_i": 0.406e-3, "rho_s": 444,
        "locator": lambda x: x[1] >= 0.1},
    "bottom": {"d_i": 1.463e-3, "rho_s": 484,
        "locator": lambda x: x[1] < 0.1},
        }

def ini_hw(x):
    return np.where(x[1] >= 0.1, -0.32, -0.175)
def ini_phi(x):
    return np.where(x[1] >= 0.1, 0.517, 0.473)

    
initial_cond = {"h_w": ini_hw,
                "phi": ini_phi,
                }


filename = "Test_Avanzi_FM1_highres"
solve_system(geom, delta_x, boundaries, bc_dict, initial_cond, layer_params=layer_params, delta_t=1e-2, T_end=90*60, saving_interval=60, save_tmp=True, filename=filename, )



