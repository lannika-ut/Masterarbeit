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
from dolfinx import mesh
from ufl import (
    grad, dx, dot,
    SpatialCoordinate, TestFunction, TrialFunction,
    rhs, lhs, system,
)
from petsc4py import PETSc
import pickle

def solve_Richards(
        h_w, h_w_old, snes, problem, b, J, delta_t, t, tmp, filename):
    min_dt = 1e-2
    new_dt = delta_t.value
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
        if float(delta_t.value) <= min_dt:
            # Save data so far:
            tmp["h_w"].append(h_w_old.x.array.copy())
            foldername = "./Masterarbeit/solutions/debug/"
            with open(foldername + filename + ".pkl", "wb") as f:
                pickle.dump(tmp, f)
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

def apply_initial_condition(f, ini):
    if callable(ini):
        f.interpolate(ini)
    else:
        f.x.array[:] = ini*np.ones_like(f.x.array)
    return f

def solve_system(
        filename, geom, delta_x, boundaries, bc_dict, initial_conditions, layer_params=None, delta_t = 0.5, T_end=24*60*60):
    # Set up domain, fem structure
    nx = int(geom.length/delta_x)
    nz = int(geom.height/delta_x)
    print(f"Resolution is dx = dz = {delta_x} m, giving nx = {nx}, nz = {nz}")
    domain = geom.make_domain(nx, nz)
    tdim = domain.topology.dim
    V_hw = functionspace(domain, ("CG", 1))
    v_hw = TestFunction(V_hw)
    V_Ti = functionspace(domain, ("CG", 1))
    v_Ti = TestFunction(V_Ti)
    V_Tw = functionspace(domain, ("CG", 1))
    v_Tw = TestFunction(V_Tw)
    Q = functionspace(domain, ("DG", 0))
    x = SpatialCoordinate(domain)

    # Parameters
    p = Parameter(domain, layer_params)

    # Set up functions, initial conditions
    h_w_old = Function(V_hw)
    h_w_old.name = "h_w_old"
    phi_old = Function(Q)
    phi_old.name = "phi_old"
    T_i_old = Function(V_Ti)
    T_i_old.name = "T_i_old"
    T_w_old = Function(V_Tw)
    T_w_old.name = "T_w_old"
    for key, ini in initial_conditions.items():
        if key == "h_w":
            h_w_old = apply_initial_condition(h_w_old, ini)
        elif key == "phi":
            phi_old = apply_initial_condition(phi_old, ini)
        elif key == "T_i":
            T_i_old = apply_initial_condition(T_i_old, ini)
        elif key == "T_w":
            T_w_old = apply_initial_condition(T_w_old, ini)
    
    # Trial Functions
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

    # Boundary conditions
    bc = BoundaryCondition(domain, boundaries)
    for key, d in bc_dict.items():
        if d["variable"] == "h_w":
            d["functionspace"] = V_hw
            d["testfunction"] = v_hw
        elif d["variable"] == "T_i":
            d["functionspace"] = V_Ti
            d["testfunction"] = v_Ti
        elif d["variable"] == "T_w":
            d["functionspace"] = V_Tw
            d["testfunction"] = v_Tw

    bcs = bc.make_boundary_condition(bc_dict)
    bc_D_hw = []
    bc_D_Ti = []
    bc_D_Tw = []
    for key, d in bc_dict.items():
        if d["name"] == "Neumann":
            # update weak formulations with Neumann bc
            if d["variable"] == "h_w":
                F_hw1 += bcs[key]
                F_hw2 += bcs[key]
            elif d["variable"] == "T_i":
                F_Ti += bcs[key]
            elif d["variable"] == "T_w":
                F_Tw += bcs[key]
        elif d["name"] == "Dirichlet":
            # sort Dirichlet bc after variable
            if d["variable"] == "h_w":
                bc_D_hw.append(bcs[key])
            elif d["variable"] == "T_i":
                bc_D_Ti.append(bcs[key])
            elif d["variable"] == "T_w":
                bc_D_Tw.append(bcs[key])
    

    # und weiter gehts