from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc


class Parameter:
    """Parametrizations used to solve the non-isothermal snowpack lateral flow model.
    """

    def __init__(self, domain, layer_params=None):
        """Construct an instance of the class Parameter.

        Args:
            domain (Dolfinx.mesh): Finite element mesh of the model.
            layer_params (dict, optional): Dictionnary containing the snow parameters rho_s (density of snow) and the ice grain diameter d_i. It also contains a boolean function to locate the layer within the domain named locator. Defaults to None.
        """
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
        Rm = self.c_pw.value / (self.beta_sol.value*self.L_sol.value)
        self.R_m = fem.Constant(domain, PETSc.ScalarType(Rm))
        Di = self.K_i.value / (self.rho_i.value*self.c_pi.value)
        Dw = self.K_w.value / (self.rho_w.value*self.c_pw.value)
        self.D_i = fem.Constant(domain, PETSc.ScalarType(Di))
        self.D_w = fem.Constant(domain, PETSc.ScalarType(Dw))

        # snow/van Genuchten parameters
        if layer_params is not None:
            param_fct = self.assign_material(domain, layer_params)
            self.d_i = param_fct["d_i"]
            self.rho_s = param_fct["rho_s"]
            self.r_i = param_fct["r_i"]
            self.r_w = param_fct["r_w"]
            self.alpha = param_fct["alpha"]
            self.N = param_fct["N"]
        else: # homogeneous snow
            self.d_i = fem.Constant(domain, PETSc.ScalarType(3e-4)) # m
            self.r_i = fem.Constant(domain, 
                                    PETSc.ScalarType(0.06*self.d_i.value)) # m
            self.r_w = fem.Constant(domain, 
                                    PETSc.ScalarType(1.35*self.d_i.value)) # m
            self.rho_s = fem.Constant(domain, PETSc.ScalarType(350)) # kg/m^3
            a = (4.4e6) * (self.rho_s.value/self.d_i.value)**(-0.98) # 1/m
            n = 1 + (2.7e-3) * (self.rho_s.value/self.d_i.value)**(0.61)
            self.alpha = fem.Constant(domain, PETSc.ScalarType(a))
            self.N = fem.Constant(domain, PETSc.ScalarType(n))
        
        self.theta_r = fem.Constant(domain, PETSc.ScalarType(0.02))
        self.SSA_0 = fem.Constant(domain, PETSc.ScalarType(4114)) # 1/m

    
    
    def assign_material(self, domain, layer_params):
        """Assign material properties to functions to account for different layer properties.

        Args:
            domain (dolfinx.mesh.Mesh): FEM domain
            layer_params (dict): Dictionnary containing the snow parameters rho_s (density of snow) and the ice grain diameter d_i. It also contains a boolean function to locate the layer within the domain named locator.
        Returns:
            dict: dictionnary of the parameter functions of d_i, rho_s, r_i=0.06*d_i, r_w=1.35*d_i and van Genuchten parameters alpha, N using the parametrization from Yamaguchi et al. (2012).
        """
        Q = fem.functionspace(domain, ("DG", 0))
        # create parameter functions
        d_i = fem.Function(Q)
        rho_s = fem.Function(Q)
        alpha = fem.Function(Q)
        N = fem.Function(Q)
        r_i = fem.Function(Q)
        r_w = fem.Function(Q)

        tdim = domain.topology.dim
        num_cells = domain.topology.index_map(tdim).size_local
        cells = np.arange(num_cells, dtype=np.int32)
        midpoints = domain.compute_midpoints(domain, tdim, cells)

        d_i_vals = np.zeros(num_cells)
        rho_s_vals = np.zeros(num_cells)
        alpha_vals = np.zeros(num_cells)
        n_vals = np.zeros(num_cells)
        ri_vals = np.zeros(num_cells)
        rw_vals = np.zeros(num_cells)

        for c, x in enumerate(midpoints):
            # Check in which layer the midpoint is and assign the corresponding parameters
            for value in layer_params.values():
                if value["locator"](x):
                    d_i_vals[c] = value["d_i"]
                    rho_s_vals[c] = value["rho_s"]
                    alpha_vals[c] = (4.4e6) * (
                        (value["rho_s"]/value["d_i"])**(-0.98))
                    n_vals[c] = 1 + (2.7e-3) * (
                        (value["rho_s"]/value["d_i"])**(0.61))
                    ri_vals[c] = 0.06*value["d_i"]
                    rw_vals[c] = 1.35*value["d_i"]

        # Fill functions with right values
        alpha.x.array[:] = alpha_vals
        N.x.array[:] = n_vals
        d_i.x.array[:] = d_i_vals
        rho_s.x.array[:] = rho_s_vals
        r_i.x.array[:] = ri_vals
        r_w.x.array[:] = rw_vals 

        alpha.x.scatter_forward()
        N.x.scatter_forward()
        d_i.x.scatter_forward()
        rho_s.x.scatter_forward()
        r_i.x.scatter_forward()
        r_w.x.scatter_forward()
        # arange into dict
        _dict = {"d_i": d_i,
                 "rho_s": rho_s,
                 "alpha": alpha,
                 "N": N,
                 "r_i": r_i,
                 "r_w": r_w
                }
        return _dict

    def S_e(self, h_w):
        """Calculate the effective saturation after van Genuchten."""
        return ufl.conditional(
            h_w < 0, 
            (1 + (-self.alpha*h_w)**self.N) ** ((1-self.N)/self.N),
            1)
    
    def theta(self, Se, phi):
        """Calculate the volumetric water content after van Genuchten."""
        return self.theta_r + (0.9*phi-self.theta_r)*Se
    
    def k_rel(self, Se):
        """Calculate the relative permeability after van Genuchten."""
        m = 1 - 1/self.N
        return ufl.conditional(
            Se < 1 - 1e-7,
            ufl.sqrt(Se) * (1 - (1-Se**(1/m))**m)**2,
            1)
    
    def K_s(self, phi):
        """Calculate the saturated hydraulic conductivity after Calonne."""
        return (3*(self.d_i/2)**2
                * ufl.exp(-0.013*self.rho_i*(1-phi))
                * self.rho_w*self.g/self.mu_w)
    
    def T_int(self, T_i, T_w, T_intold):
        """Calculate the interface temperature after Moure et al. (2023)."""
        rho = ufl.conditional(T_w < T_intold, self.rho_w, self.rho_i)
        numerator = (self.K_i/self.r_i*T_i
                     + self.K_w/self.r_w*T_w
                     + rho*self.L_sol*self.R_m*self.T_melt)
        denominator = (self.K_i/self.r_i
                       + self.K_w/self.r_w
                       + rho*self.L_sol* self.R_m)
        return numerator/denominator
    
    def W_SSA(self, Se, phi):
        """Calculate the wet specific surface area. """
        part1 = self.theta(Se, phi) * ufl.ln(phi)
        phi0 = 1 - self.rho_s/self.rho_i
        return part1*self.SSA_0 / (phi0*ufl.ln(phi0))

""" from dolfinx import mesh

msh = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

V = fem.functionspace(msh, ("CG", 1))
h_w = fem.Function(V)
h_w.x.array[:] = -0.1
phi = fem.Function(V)
phi.x.array[:] = 0.4
layers = {1: {"d_i": 0.3e-3, "rho_s": 363, "locator": lambda x: True}}
p = Parameter(domain=msh, layer_params=layers)
S = p.S_e(h_w)
t = p.T_int(S, phi, 15)
print(type(t)) """