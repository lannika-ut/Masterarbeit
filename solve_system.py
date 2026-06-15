import numpy as np
from parametrizations import Parameter
from boundary_condition import BoundaryCondition
from geometry_class import Geometry
from nonlinear_snes_problem import NonlinearPDE_SNESProblem
from dolfinx import fem
from dolfinx.fem.petsc import create_matrix, create_vector
from ufl import (
    grad, dx, dot, SpatialCoordinate, TestFunction, TrialFunction,
)
from petsc4py import PETSc

# Set up geometry
height = 2
length = 1
delta_x = delta_z = 0.05
nx = int(6/delta_x)
nz = int(3/delta_x)
print(
    f"Resolution is dx = {delta_x} m, dz = {delta_z} m, giving nx = {nx}, nz = {nz}")

geom = Geometry(height, length, slope=0)
domain = geom.make_domain(nx, nz)
V = fem.functionspace(domain, ("CG", 1))
x = SpatialCoordinate(domain)
v = TestFunction(V)

# Get parameters
p = Parameter(domain)

# Set up boundary conditions
boundaries = {
    1: lambda x: np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1)),
    2: lambda x: np.isclose(x[1], 2),
    3: lambda x: np.isclose(x[1], 0)}
bc_dict = {
    "top_hw": {
        "marker": 2, "name": "Dirichlet", "value": 1,
        "functionspace": V, "testfunction": v},
    "top_Ti": {
        "marker": 2, "name": "Dirichlet", "value": 0,
        "functionspace": V, "testfunction": v},
    "top_Tw": {
        "marker": 2, "name": "Dirichlet", "value": 0,
        "functionspace": V, "testfunction": v},
    "bottom_Ti": {
        "marker": 3, "name": "Dirichlet", "value": -5,
        "functionspace": V, "testfunction": v}
}

bc = BoundaryCondition(domain, boundaries)
bcs = bc.make_boundary_condition(bc_dict)

# Set up time iteration
delta_t = fem.Constant(domain, PETSc.ScalarType(7))
T_end = 24*60*60
t = 0.0

# Create Newton solver
snes = PETSc.SNES().create()

# Initial conditions
h_w_old = fem.Function(V)
h_w_old.name = "h_w_old"
h_w_old.x.array[:] = -0.3*np.ones_like(h_w_old.x.array)

phi_old = fem.Function(V)
phi_old.name = "phi_old"
phi_old.x.array[:] = 0.468

T_i_old = fem.Function(V)
T_i_old.name = "T_i_old"
T_i_old.x.array[:] = -5

T_w_old = fem.Function(V)
T_w_old.name = "T_w_old"
T_w_old.x.array[:] = 0

T_intold = p.T_int(T_i_old, T_w_old, -1)  # ????

# Trial functions
h_w = fem.Function(V)
phi = fem.Function(V)
T_i = fem.Function(V)
T_w = fem.Function(V)

# Weak formulation
F_hw = (
    v * (p.theta(p.S_e(h_w), phi) -
         p.theta(p.S_e(h_w_old), phi_old)) / delta_t * dx
    + dot(grad(v), (p.K_s(phi_old)*p.k_rel(p.S_e(h_w))*grad(x[1]+h_w)))*dx
    - v*p.rho_i/p.rho_w*p.R_m *
        (p.T_int(T_i_old, T_w_old, T_intold) - p.T_melt) *
    p.W_SSA(p.S_e(h_w_old), phi_old)*dx
)
F_Ti = (
    v * (1-phi)*(T_i - T_i_old)/delta_t * dx
    + dot(grad(v), p.D_i*(1-phi)*grad(T_i)) * dx
    - v*p.D_i*p.W_SSA(p.S_e(h_w), phi) *
    (p.T_int(T_i_old, T_w_old, T_intold)-T_i)/p.r_i * dx
)

F_Tw = (
    v * p.theta(p.S_e(h_w), phi)*(T_w - T_w_old)/delta_t * dx
    + dot(grad(v), (p.D_w*p.theta(p.S_e(h_w), phi)*grad(T_w)
                    + p.K_s(phi)*p.k_rel(p.S_e(h_w))*grad(x[1]+h_w)*T_w)) * dx
    - v*p.D_w*p.W_SSA(p.S_e(h_w), phi) *
    (p.T_int(T_i_old, T_w_old, T_intold) - T_w)/p.r_w * dx
)

problem_hw = NonlinearPDE_SNESProblem(F_hw, h_w, bcs["top_hw"])
b_hw = create_vector(V)
J_hw = create_matrix(problem_hw.a)

snes.destroy()
