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
    min_dt = 1e-2
    new_dt = delta_t.value
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

    converged = snes.getConvergedReason()
    num_iter = snes.getIterationNumber()
    if converged <= 0:
        if float(delta_t.value) <= min_dt:
            raise RuntimeError(
                f"Solver failed to converge (reason {converged}) even at "
                f"the minimum time step {min_dt} s, t={t/3600:.2f} h."
            )
        new_dt = max(0.5*float(delta_t.value), min_dt)
        print(f"Newton diverged (reason {converged}), halving dt to "
              f"{delta_t.value:.4f} s and retrying t={t/3600:.2f} h.")
        repeat_time_step = True
        return h_w, repeat_time_step, new_dt
    
    sol_vec.copy(h_w.x.petsc_vec)  # copy solution into h_w
    h_w.x.scatter_forward()

    # Converged, but check if it was "slow" and should shrink dt anyway
    if num_iter > 10 and float(delta_t.value) > min_dt:
        new_dt = max(0.5*float(delta_t.value), min_dt)
        repeat_time_step = True
        return h_w, repeat_time_step, new_dt

    if num_iter < 3 and float(delta_t.value) < 7:
        new_dt = min(float(delta_t.value)*1.2, 7)

    print(
        f"Solver converged after {num_iter} iterations, reason {converged}. "
        f"dt = {delta_t.value:.2f} s at t={t/3600:.2f} h."
    )
    return h_w, repeat_time_step, new_dt


def validate_state(h_w, phi, T_i, T_w, label="State"):
    print(f"\n{label} validation:")
    print(f"  h_w: min={h_w.x.array.min():.3e}, max={h_w.x.array.max():.3e}")
    print(f"  phi: min={phi.x.array.min():.3f}, max={phi.x.array.max():.3f}")
    print(f"  T_i: min={T_i.x.array.min():.3f}, max={T_i.x.array.max():.3f}")
    print(f"  T_w: min={T_w.x.array.min():.3f}, max={T_w.x.array.max():.3f}")
    assert not np.any(np.isnan(h_w.x.array)), "NaN in h_w"
    assert not np.any(np.isnan(phi.x.array)), "NaN in phi"
    assert not np.any(np.isnan(T_i.x.array)), "NaN in T_i"
    assert not np.any(np.isnan(T_w.x.array)), "NaN in T_w"


# Set up geometry
height = 2
length = 1
delta_x = delta_z = 0.1
nx = int(6/delta_x)
nz = int(3/delta_x)
print(
    f"Resolution is dx = {delta_x} m, dz = {delta_z} m, giving nx = {nx}, nz = {nz}")

geom = Geometry(height, length, slope=0)
domain = geom.make_domain(nx, nz)
V_hw = functionspace(domain, ("CG", 1))
v_hw = TestFunction(V_hw)
V_Ti = functionspace(domain, ("CG", 1))
v_Ti = TestFunction(V_Ti)
V_Tw = functionspace(domain, ("CG", 1))
v_Tw = TestFunction(V_Tw)
Q = functionspace(domain, ("DG", 0))
x = SpatialCoordinate(domain)

# Get parameters
p = Parameter(domain)

# Set up boundary conditions
boundaries = {
    1: lambda x: np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1)),
    2: lambda x: np.isclose(x[1], 2),
    3: lambda x: np.isclose(x[1], 0)}
bc_dict = {
    "top_Ti": {
        "marker": 2, "name": "Dirichlet", "value": 0,
        "functionspace": V_Ti, "testfunction": v_Ti},
    "top_Tw": {
        "marker": 2, "name": "Dirichlet", "value": 0,
        "functionspace": V_Tw, "testfunction": v_Tw},
    "bottom_Ti": {
        "marker": 3, "name": "Dirichlet", "value": -1,
        "functionspace": V_Ti, "testfunction": v_Ti}
}

bc = BoundaryCondition(domain, boundaries)
bcs, dontuse = bc.make_boundary_condition(bc_dict)
print(bcs)

# Set up time iteration
delta_t = Constant(domain, PETSc.ScalarType(0.5))
T_end = 24*60*60
t = 0.0

# Initial conditions
h_w_old = Function(V_hw)
h_w_old.name = "h_w_old"
h_w_old.x.array[:] = -0.22*np.ones_like(h_w_old.x.array)

phi_old = Function(Q)
phi_old.name = "phi_old"
phi_old.x.array[:] = 0.468*np.ones_like(phi_old.x.array)

T_i_old = Function(V_Ti)
T_i_old.name = "T_i_old"
def temp_gradient(x):
    return x[1]/2-1
T_i_old.interpolate(temp_gradient)

T_w_old = Function(V_Tw)
T_w_old.name = "T_w_old"
T_w_old.x.array[:] = np.zeros_like(T_w_old.x.array)


# Trial functions
h_w = Function(V_hw)
phi = Function(Q)
T_i = TrialFunction(V_Ti)
T_w = TrialFunction(V_Tw)
# Create intermediate solution functions
T_i_h = Function(V_Ti)
T_i_h.name = "T_i_h"
T_w_h = Function(V_Tw)
T_w_h.name = "T_w_h"
h_w1 = Function(V_hw)
h_w1.name = "h_w1"
phi1 = Function(Q)
phi1.name = "phi1"
# DG0 function needed for evaluation
krel = Function(Q)
krel.name = "krel"
source_mass = Function(Q)
source_mass.name = "source_mass"


# Weak formulation
F_hw1 = (
    v_hw * (p.theta(p.S_e(h_w), phi1) -
            p.theta(p.S_e(h_w_old), phi_old)) / delta_t * dx
    + dot(grad(v_hw), (p.K_s(phi1)*krel*grad(x[1]+h_w)))*dx
    - v_hw*p.rho_i/p.rho_w*source_mass*dx
)

F_hw2 = (
    v_hw * (p.theta(p.S_e(h_w), phi) -
            p.theta(p.S_e(h_w_old), phi_old)) / delta_t * dx
    + dot(grad(v_hw), (p.K_s(phi)*krel*grad(x[1]+h_w)))*dx
    - v_hw*p.rho_i/p.rho_w*source_mass*dx
)

F_Ti = (
    v_Ti * (1-phi1)*(T_i - T_i_old)/delta_t * dx
    + dot(grad(v_Ti), p.D_i*(1-phi1)*grad(T_i)) * dx
    - v_Ti*p.D_i*p.W_SSA(p.S_e(h_w1), phi1) *
    (p.T_int(T_i_old, T_w_old)-T_i)/p.r_i * dx
)

F_Tw = (
    v_Tw * p.theta(p.S_e(h_w1), phi1)*(T_w - T_w_old)/delta_t * dx
    + dot(grad(v_Tw), (p.D_w*p.theta(p.S_e(h_w1), phi1)*grad(T_w)
                       + p.K_s(phi1)*krel*grad(x[1]+h_w1)*T_w)) * dx
    - v_Tw*p.D_w*p.W_SSA(p.S_e(h_w1), phi1) *
    (p.T_int(T_i_old, T_w_old) - T_w)/p.r_w * dx
)

# Create Newton solver
snes1 = PETSc.SNES().create()
snes2 = PETSc.SNES().create()
# Set up nonlinear problem
problem_hw1 = NonlinearPDE_SNESProblem(F_hw1, h_w, bc=None)
b_hw1 = create_vector(V_hw)
J_hw1 = create_matrix(problem_hw1.a)
problem_hw2 = NonlinearPDE_SNESProblem(F_hw2, h_w, bc=None)
b_hw2 = create_vector(V_hw)
J_hw2 = create_matrix(problem_hw2.a)

# Set up linear solver options
petsc_options = {
    "ksp_error_if_not_converged": True,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-4,
    "ksp_atol": 1e-6,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_max_iter": 1,
    "pc_hypre_boomeramg_cycle_type": "v",
}
# Set up linear problem
a_Ti, L_Ti = system(F_Ti)
a_Tw, L_Tw = system(F_Tw)

# Create structure for saving intermediate results
tmp = {
    "geometry": geom.make_into_dict(),
    "T_end": T_end,
    "parameter": p.make_into_dict(),
    "h_w": [],
    "phi": [],
    "T_i": [],
    "T_w": [],
    "T_int": [],
    "times": [],
    "k_rel":[],
}
tmp["h_w"].append(h_w_old.x.array.copy())
tmp["phi"].append(phi_old.x.array.copy())
tmp["T_i"].append(T_i_old.x.array.copy())
tmp["T_w"].append(T_w_old.x.array.copy())
tmp["times"].append(t)
tmp["k_rel"].append(krel.x.array.copy())
next_saving_time = 30*60

# Time loop
while t <= T_end:
    
    # Upwind krel
    new_krel = p.upwind_krel(h_w_old, domain)
    krel.x.array[:] = new_krel.x.array
    krel.x.scatter_forward()

    # Calculate source term
    new_source = p.calc_source_term(h_w_old, phi_old, T_i_old, T_w_old)
    source_mass.x.array[:] = new_source.x.array
    source_mass.x.scatter_forward()

    # Update porosity
    max_source = 0.1 * phi_old.x.array / delta_t.value  # Limit to 10% change per step
    source_term = np.clip(source_mass.x.array, -max_source, max_source)
    phi1.x.array[:] = phi_old.x.array + delta_t.value * source_term
    # Ensure porosity stays between 0 and 1
    phi1.x.array[:] = np.clip(phi1.x.array, 0, 1)

    # Solve Richards
    h_w1, repeat_time_step, new_dt = solve_Richards(
        h_w, h_w_old, snes1, problem_hw1, b_hw1, J_hw1, delta_t, t)
    if repeat_time_step:
        delta_t.value = new_dt
        continue
    
    # Update krel with new pressure head
    new_krel = p.upwind_krel(h_w1, domain)
    krel.x.array[:] = new_krel.x.array
    krel.x.scatter_forward()
    # Solve Thermodynamics
    problem_Ti = LinearProblem(
        a_Ti, L_Ti, bcs=[bcs["top_Ti"], bcs["bottom_Ti"]], 
        petsc_options=petsc_options, petsc_options_prefix="T_i")
    T_i_h = problem_Ti.solve()
    T_i_old.x.array[:] = T_i_h.x.array
    problem_Tw = LinearProblem(
        a_Tw, L_Tw, bcs=[bcs["top_Tw"]],
        petsc_options=petsc_options, petsc_options_prefix="T_w")
    T_w_h = problem_Tw.solve()
    T_w_old.x.array[:] = T_w_h.x.array

    # Update source term with new values
    new_source = p.calc_source_term(h_w1, phi1, T_i_h, T_w_h)
    source_mass.x.array[:] = new_source.x.array
    source_mass.x.scatter_forward()

    # Update porosity again 
    source_term = np.clip(source_mass.x.array, -max_source, max_source)
    phi.x.array[:] = phi_old.x.array + delta_t.value * source_term
    # Ensure porosity stays between 0 and 1
    phi.x.array[:] = np.clip(phi.x.array, 0, 1)

    # Solve Richards again
    h_w, repeat_time_step, dontusethistimestep = solve_Richards(
        h_w, h_w_old, snes2, problem_hw2, b_hw2, J_hw2, delta_t, t)

    # save temporary data
    if True and t >= next_saving_time:
        next_saving_time += 30*60
        tmp["h_w"].append(h_w.x.array.copy())
        tmp["phi"].append(phi.x.array.copy())
        tmp["T_i"].append(T_i_old.x.array.copy())
        tmp["T_w"].append(T_w_old.x.array.copy())
        tmp["k_rel"].append(krel.x.array.copy())
        tmp["times"].append(t)

    h_w_old.x.array[:] = h_w.x.array
    phi_old.x.array[:] = phi.x.array
    #validate_state(h_w_old, phi_old, T_i_old, T_w_old, label="After correction")
    delta_t.value = new_dt
    t += float(delta_t.value)


# Destroy PETSc objects
snes1.destroy()
snes2.destroy()
b_hw1.destroy()
J_hw1.destroy()
b_hw2.destroy()
J_hw2.destroy()

# save final data
tmp["h_w"].append(h_w.x.array.copy())
tmp["phi"].append(phi.x.array.copy())
tmp["T_i"].append(T_i_old.x.array.copy())
tmp["T_w"].append(T_w_old.x.array.copy())
tmp["k_rel"].append(krel.x.array.copy())
tmp["times"].append(t)
filename = "./Masterarbeit/solutions/full_1day_upwind.pkl"
with open(filename, "wb") as f:
    pickle.dump(tmp, f)
