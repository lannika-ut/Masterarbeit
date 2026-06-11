from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc


class parameter:

    def __init__(self, domain):
        self.rho_i = fem.Constant(domain, PETSc.ScalarType(917)) # kg/m^3
        self.rho_w = fem.Constant(domain, PETSc.ScalarType(1000)) # kg/m^3
        self.mu_w = fem.Constant(domain, PETSc.ScalarType(1.7e-3)) # kg/(m*s)
        self.g = fem.Constant(domain, PETSc.ScalarType(9.81)) # m/s^2

        # Thermal properties
        self.c_pw = fem.Constant(domain, PETSc.ScalarType(4200)) # J/(kg*K)
        self.c_pi = fem.Constant(domain, PETSc.ScalarType(2040)) # J/(kg*K)
        self.K_i = fem.Constant(domain, PETSc.ScalarType(2.2)) # W/(m*K)
        self.K_w = fem.Constant(domain, PETSc.ScalarType(0.55)) # W/(m*K)
        self.T_melt = fem.Constant(domain, PETSc.ScalarType(273.15)) # K
        self.L_sol = fem.Constant(domain, PETSc.ScalarType(3.34e5)) # J/kg
        self.beta_sol = fem.Constant(domain, PETSc.ScalarType(800)) # s/m

        # Composite parameters
        Rm = self.c_pw.value/(self.beta_sol.value * self.L_sol.value)
        self.R_m = fem.Constant(domain, PETSc.ScalarType(Rm))
        Di = self.K_i.value / (self.rho_i.value * self.c_pi.value)
        Dw = self.K_w.value / (self.rho_w.value * self.c_pw.value)
        self.D_i = fem.Constant(domain, PETSc.ScalarType(Di))
        self.D_w = fem.Constant(domain, PETSc.ScalarType(Dw))


from dolfinx import mesh

msh = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
p = parameter(domain=msh)
rm = p.K_i.value/(p.rho_i.value*p.c_pi.value)
print(rm)
print(p.D_i.value)