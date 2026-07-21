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
    grad, dx, dot, outer,
    SpatialCoordinate, TestFunction, TrialFunction,
    rhs, lhs, system,
)
from petsc4py import PETSc
import pickle

def solve_Richards(
        h_w, h_w_old, snes, problem, b, J, delta_t, t, tmp, filename, phi, Ti, Tw):
    min_dt = 1e-2
    max_dt = 1
    new_dt = delta_t.value
    repeat_time_step = False
    h_w.x.array[:] = h_w_old.x.array
    snes.setFunction(problem.F, b)  # assemble residual
    snes.setJacobian(problem.J, J)  # assemble Jacobian

    # Set options
    snes.setType("newtonls")
    snes.getLineSearch().setType(PETSc.SNESLineSearch.Type.BT)
    snes.setTolerances(rtol=1e-4, atol=1e-4, max_it=50)
    ksp = snes.getKSP()
    ksp.setType("gmres")  # iterative solver
    ksp.setTolerances(rtol=1e-4)
    ksp.setErrorIfNotConverged(False)
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
            tmp["phi"].append(phi.x.array.copy())
            tmp["T_i"].append(Ti.x.array.copy())
            tmp["T_w"].append(Tw.x.array.copy())
            tmp["times"].append(t)
            foldername = "./Masterarbeit/solutions/debug/"
            with open(foldername + filename + ".pkl", "wb") as f:
                pickle.dump(tmp, f)
            raise RuntimeError(
                f"Solver failed to converge (reason {converged}) even at "
                f"the minimum time step {min_dt} s, t={t/3600:.4f} h."
            )
        new_dt = max(0.5*float(delta_t.value), min_dt)
        print(f"Newton diverged (reason {converged}), halving dt to "
              f"{new_dt:.3f} s and retrying t={t/3600:.4f} h.")
        repeat_time_step = True
        return sol_vec, repeat_time_step, new_dt
    # Converged, but check if it was "slow" and should shrink dt anyway
    if num_iter > 10 and float(delta_t.value) > min_dt:
        new_dt = max(0.5*float(delta_t.value), min_dt)
        repeat_time_step = True
        return sol_vec, repeat_time_step, new_dt
    if num_iter < 3 and float(delta_t.value) < max_dt:
        new_dt = min(float(delta_t.value)*1.2, max_dt)
    print(
        f"Solver converged after {num_iter} iterations, reason {converged}. "
        f"dt = {delta_t.value:.3f} s at t={t/3600:.4f} h.")
    return sol_vec, repeat_time_step, new_dt

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
        filename, geom, delta_x, boundaries, bc_dict, initial_conditions, layer_params=None, delta_t = 0.5, T_end=24*60*60, saving_interval=60):
    """Solve the full system with mass conservation and local thermal non-equilibrium.

    Args:
        filename (string): Filename (and foldername) where results should be saved to.
        geom (Geometry): The geometry of the domain (includes height, length, slope).
        delta_x (float): The resolution.
        boundaries (_type_): _description_
        bc_dict (_type_): _description_
        initial_conditions (_type_): _description_
        layer_params (_type_, optional): _description_. Defaults to None.
        delta_t (float, optional): _description_. Defaults to 0.5.
        T_end (_type_, optional): _description_. Defaults to 24*60*60.
        saving_interval (int, optional): _description_. Defaults to 60.
    """
    # Set up domain, fem structure
    nx = int(geom.length/delta_x)
    nz = int(geom.height/delta_x)
    print(f"Resolution is dx = dz = {delta_x} m, giving nx = {nx}, nz = {nz}")
    domain = geom.make_domain(nx, nz, celltype=mesh.CellType.quadrilateral)
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
    # Time parameters
    delta_t = Constant(domain, PETSc.ScalarType(delta_t))
    t = 0

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
    tau = Constant(domain, PETSc.ScalarType(0.01))
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
    q1 = p.K_s(phi1)*krel*grad(x[1]+h_w1)
    eps = 10*np.finfo(np.float64).eps
    weights_sum = (p.c_pw/p.L_sol
                   + p.beta_sol/(p.rho_w*p.L_sol*p.r_i)
                   + p.beta_sol/(p.rho_w*p.L_sol*p.r_w))
    a_i = (p.beta_sol/(p.rho_w*p.L_sol*p.r_i))/weights_sum
    a_w = (p.beta_sol/(p.rho_w*p.L_sol*p.r_w))/weights_sum
    F_Ti = (
        v_Ti * (1-phi1)*(T_i - T_i_old)/delta_t * dx
        + dot(grad(v_Ti), p.D_i*(1-phi1)*grad(T_i)) * dx
        - v_Ti*p.D_i*p.W_SSA(p.S_e(h_w1), phi1) *
        ((a_i-1)*T_i + a_w*T_w_h)/p.r_i * dx
        + dot(grad(v_Ti), tau/(dot(q1,q1)+eps)*outer(q1,q1)*grad(T_i)) * dx # artificial diffusion
    )
    F_Tw = (
        v_Tw * p.theta(p.S_e(h_w1), phi1)*(T_w - T_w_old)/delta_t * dx
        + dot(grad(v_Tw), (p.D_w*p.theta(p.S_e(h_w1), phi1)*grad(T_w)
                        + p.K_s(phi1)*krel*grad(x[1]+h_w1)*T_w)) * dx
        - v_Tw*p.D_w*p.W_SSA(p.S_e(h_w1), phi1) *
        (a_i*T_i_h + (a_w-1)*T_w)/p.r_w * dx
        + dot(grad(v_Tw), tau/(dot(q1,q1)+eps)*outer(q1,q1)*grad(T_w)) * dx # artificial diffusion
    )

    # Boundary conditions
    print_bc = bc_dict.copy()
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
    
    # Create solver structure
    snes1 = PETSc.SNES().create()
    snes2 = PETSc.SNES().create()
    # Set up nonlinear problem
    problem_hw1 = NonlinearPDE_SNESProblem(F_hw1, h_w, bc=bc_D_hw)
    b_hw1 = create_vector(V_hw)
    J_hw1 = create_matrix(problem_hw1.a)
    problem_hw2 = NonlinearPDE_SNESProblem(F_hw2, h_w, bc=bc_D_hw)
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
        "saving_interval": saving_interval,
        "boundary_condition": str(print_bc),
        "initial_condition": str(initial_conditions),
        "tau": tau.value
    }
    tmp["h_w"].append(h_w_old.x.array.copy())
    tmp["phi"].append(phi_old.x.array.copy())
    tmp["T_i"].append(T_i_old.x.array.copy())
    tmp["T_w"].append(T_w_old.x.array.copy())
    tmp["times"].append(t)
    tmp["k_rel"].append(krel.x.array.copy())
    next_saving_time = saving_interval

    # Time loop
    while t <= T_end:
        # Upwind krel
        new_krel = p.upwind_krel(h_w_old, domain)
        krel.x.array[:] = new_krel.x.array.copy()
        krel.x.scatter_forward()

        # Calculate source term
        new_source = p.calc_source_term(h_w_old, phi_old, T_i_old, T_w_old)
        source_mass.x.array[:] = new_source.x.array.copy()
        source_mass.x.scatter_forward()

        # Update porosity
        max_source = 0.1 * phi_old.x.array / delta_t.value  # Limit to 10% change per step
        source_term = np.clip(source_mass.x.array, -max_source, max_source)
        phi1.x.array[:] = phi_old.x.array + delta_t.value * source_term
        phi1.x.array[:] = np.clip(phi1.x.array, 0, 1)

        # Solve Richards
        sol_vec, repeat_time_step, new_dt = solve_Richards(
            h_w, h_w_old, snes1, problem_hw1, b_hw1, J_hw1, delta_t, t, tmp, filename, phi1, T_i_old, T_w_old)
        if repeat_time_step:
            delta_t.value = new_dt
            continue
        sol_vec.copy(h_w1.x.petsc_vec)  # copy solution into h_w1
        h_w1.x.scatter_forward()
        
        # Update krel with new pressure head
        new_krel = p.upwind_krel(h_w1, domain)
        krel.x.array[:] = new_krel.x.array.copy()
        krel.x.scatter_forward()

        #print("krel min/max:",
        #np.min(krel.x.array),
        #np.max(krel.x.array))

        # Solve Thermodynamics in Picard Loop
        T_i_h.x.array[:] = T_i_old.x.array
        T_w_h.x.array[:] = T_w_old.x.array
        for k in range(5):
            Ti_old_picard = T_i_h.x.array.copy()
            Tw_old_picard = T_w_h.x.array.copy()
            problem_Tw = LinearProblem(
                a_Tw, L_Tw, bcs=bc_D_Tw, 
                petsc_options=petsc_options, petsc_options_prefix="T_w")
            T_w_new = problem_Tw.solve()
            T_w_h.x.array[:] = T_w_new.x.array.copy()
            T_w_h.x.scatter_forward()
            problem_Ti = LinearProblem(
                a_Ti, L_Ti, bcs=bc_D_Ti, 
                petsc_options=petsc_options, petsc_options_prefix="T_i")
            T_i_new = problem_Ti.solve()
            T_i_h.x.array[:] = T_i_new.x.array.copy()
            T_i_h.x.scatter_forward()
            #err_i = np.max(abs(T_i_h.x.array - Ti_old_picard))
            #err_w = np.max(abs(T_w_h.x.array - Tw_old_picard))
            #print(k, err_i, err_w)
            k += 1
        # Update temperatures
        T_i_old.x.array[:] = T_i_h.x.array.copy()
        T_w_old.x.array[:] = T_w_h.x.array.copy()

        # Update source term with new values
        new_source = p.calc_source_term(h_w1, phi1, T_i_h, T_w_h)
        source_mass.x.array[:] = new_source.x.array.copy()
        source_mass.x.scatter_forward()

        # Update porosity again 
        source_term = np.clip(source_mass.x.array, -max_source, max_source)
        phi.x.array[:] = phi_old.x.array + delta_t.value * source_term
        phi.x.array[:] = np.clip(phi.x.array, 0, 1)

        # Solve Richards again
        sol_vec, repeat_time_step, dontusethistimestep = solve_Richards(
            h_w, h_w_old, snes2, problem_hw2, b_hw2, J_hw2, delta_t, t, tmp, filename, phi, T_i_h, T_w_h)
        sol_vec.copy(h_w.x.petsc_vec)  # copy solution into h_w
        h_w.x.scatter_forward()

        # Debug
        #print("-----")
        #print(np.max(np.abs(h_w1.x.array)))
        #print(np.max(np.abs(phi1.x.array)))
        #print(np.max(np.abs(T_i_h.x.array)))
        #print(np.max(np.abs(T_w_h.x.array)))

        # save temporary data
        if t >= next_saving_time:
            next_saving_time += saving_interval
            tmp["h_w"].append(h_w.x.array.copy())
            tmp["phi"].append(phi.x.array.copy())
            tmp["T_i"].append(T_i_old.x.array.copy())
            tmp["T_w"].append(T_w_old.x.array.copy())
            tmp["k_rel"].append(krel.x.array.copy())
            tmp["times"].append(t)

        h_w_old.x.array[:] = h_w.x.array.copy()
        phi_old.x.array[:] = phi.x.array.copy()
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
    with open("./Masterarbeit/solutions/" + filename + ".pkl", "wb") as f:
        pickle.dump(tmp, f)



# Test function
height = 2
length = 1
geom = Geometry(height, length)
boundaries = {
    1: lambda x: np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], length)), # lateral
    2: lambda x: np.isclose(x[1], height), # top
    3: lambda x: np.isclose(x[1], 0)} # bottom
bc_dict = {
    "top_Ti": {
        "marker": 2, "name": "Dirichlet", "value": 0, "variable": "T_i"},
    "top_Tw": {
        "marker": 2, "name": "Dirichlet", "value": 0, "variable": "T_w"},
    "top_hw": {
        "marker": 2, "name": "Dirichlet", "value": 0.8, "variable": "h_w"},
    "bottom_Ti": {
        "marker": 3, "name": "Dirichlet", "value": -5, "variable": "T_i"}
}

initial_cond = {"h_w": -0.22, "phi": 0.468, "T_i": -5, "T_w": 0}
solve_system("test6_crippa_uniformInfiltration", geom, 0.05, boundaries, bc_dict, initial_cond, T_end=2*60, saving_interval=1, delta_t=1e-2)
