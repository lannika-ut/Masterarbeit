from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, mesh
from petsc4py import PETSc
import inspect


class Parameter:
    """Parametrizations used to solve the non-isothermal snowpack lateral flow model.
    """

    def __init__(self, domain, layer_params=None):
        """Construct an instance of the class Parameter.

        Args:
            domain (Dolfinx.mesh): Finite element mesh of the model.
            layer_params (dict, optional): Dictionnary containing the snow parameters rho_s (density of snow) and the ice grain diameter d_i. It also contains a boolean function to locate the layer within the domain named locator. Defaults to None. 
                layer_params = {
                    key: {
                        "d_i": (float),
                        "rho_s": (float),
                        "locator": lambda x: True_if_in_layer
                        },
                        ...
                    }
        """
        self.rho_i = fem.Constant(domain, PETSc.ScalarType(917))  # kg/m^3
        self.rho_w = fem.Constant(domain, PETSc.ScalarType(1000))  # kg/m^3
        self.mu_w = fem.Constant(domain, PETSc.ScalarType(1.7e-3))  # kg/(m*s)
        self.g = fem.Constant(domain, PETSc.ScalarType(9.81))  # m/s^2

        # Thermal properties
        self.c_pw = fem.Constant(domain, PETSc.ScalarType(4200))  # J/(kg*°C)
        self.c_pi = fem.Constant(domain, PETSc.ScalarType(2040))  # J/(kg*°C)
        self.K_i = fem.Constant(domain, PETSc.ScalarType(2.2))  # W/(m*°C)
        self.K_w = fem.Constant(domain, PETSc.ScalarType(0.55))  # W/(m*°C)
        self.T_melt = fem.Constant(domain, PETSc.ScalarType(0))  # °C
        self.L_sol = fem.Constant(domain, PETSc.ScalarType(3.34e5))  # J/kg
        self.beta_sol = fem.Constant(domain, PETSc.ScalarType(800))  # s/m

        # Composite parameters
        Rm = self.c_pw.value / (self.beta_sol.value*self.L_sol.value)
        self.R_m = fem.Constant(domain, PETSc.ScalarType(Rm))  # m/(s*°C)
        Di = self.K_i.value / (self.rho_i.value*self.c_pi.value)
        Dw = self.K_w.value / (self.rho_w.value*self.c_pw.value)
        self.D_i = fem.Constant(domain, PETSc.ScalarType(Di))  # m^2/2
        self.D_w = fem.Constant(domain, PETSc.ScalarType(Dw))  # m^2/s

        self.layer_params_dict = layer_params
        # snow/van Genuchten parameters
        if layer_params is not None:
            self.is_layered = True
            param_fct = self._assign_material(domain)
            self.d_i = param_fct["d_i"]
            self.rho_s = param_fct["rho_s"]
            self.r_i = param_fct["r_i"]
            self.r_w = param_fct["r_w"]
            self.alpha = param_fct["alpha"]
            self.N = param_fct["N"]
            self.layer_params_dict = layer_params
        else:  # homogeneous snow
            self.is_layered = False
            self.d_i = fem.Constant(domain, PETSc.ScalarType(3e-4))  # m
            self.r_i = fem.Constant(domain,
                                    PETSc.ScalarType(0.06*self.d_i.value))  # m
            self.r_w = fem.Constant(domain,
                                    PETSc.ScalarType(1.35*self.d_i.value))  # m
            self.rho_s = fem.Constant(domain, PETSc.ScalarType(350))  # kg/m^3
            a = (4.4e6) * (self.rho_s.value/self.d_i.value)**(-0.98)  # 1/m
            n = 1 + (2.7e-3) * (self.rho_s.value/self.d_i.value)**(0.61)
            self.alpha = fem.Constant(domain, PETSc.ScalarType(a))
            self.N = fem.Constant(domain, PETSc.ScalarType(n))

        self.theta_r = fem.Constant(domain, PETSc.ScalarType(0.02))
        self.SSA_0 = fem.Constant(domain, PETSc.ScalarType(4114))  # 1/m
        self.S_r = fem.Constant(domain, PETSc.ScalarType(1e-3))  # residual saturation

    def _assign_material(self, domain):
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
        midpoints = mesh.compute_midpoints(domain, tdim, cells)

        d_i_vals = np.zeros(num_cells)
        rho_s_vals = np.zeros(num_cells)
        alpha_vals = np.zeros(num_cells)
        n_vals = np.zeros(num_cells)
        ri_vals = np.zeros(num_cells)
        rw_vals = np.zeros(num_cells)

        for c, x in enumerate(midpoints):
            # Check in which layer the midpoint is and assign the corresponding parameters
            for value in self.layer_params_dict.values():
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
        t = self.theta_r + (0.9*phi-self.theta_r)*Se
        return t

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

    def T_int(self, T_i, T_w, T_intold=None):
        """Calculate the interface temperature after Moure et al. (2023)."""
        # rho = ufl.conditional(T_intold < self.T_melt, self.rho_w, self.rho_i)
        rho = self.rho_w
        numerator = (self.K_i/self.r_i*T_i
                     + self.K_w/self.r_w*T_w
                     + rho*self.L_sol*self.R_m*self.T_melt)
        denominator = (self.K_i/self.r_i
                       + self.K_w/self.r_w
                       + rho*self.L_sol * self.R_m)
        return numerator/denominator

    def W_SSA(self, Se, phi):
        """Calculate the wet specific surface area. """
        part1 = (Se) * ufl.ln(phi) * phi
        phi0 = 1 - self.rho_s/self.rho_i
        return part1*self.SSA_0 / (phi0*ufl.ln(phi0))

    def S_e_numerical(self, h_w):
        """Numerical evaluation of the effective saturation after van Genuchten."""
        hw = np.array(h_w.x.array)
        Se = np.ones_like(hw)
        if self.is_layered:
            a = np.array(self.alpha.x.array)
            n = np.array(self.N.x.array)
            Se[hw < 0] = ((1 + (-a[hw < 0]*hw[hw < 0])**n[hw < 0])
                          ** ((1-n[hw < 0])/n[hw < 0]))
        else:
            a = self.alpha.value
            n = self.N.value
            Se[hw < 0] = (1 + (-a*hw[hw < 0])**n) ** ((1-n)/n)
        return Se

    def theta_numerical(self, Se, phi):
        """Numerical evaluation of the liquid water content."""
        t = (self.theta_r.value
             + (0.9*np.array(phi.x.array)-self.theta_r.value)*Se)
        return t

    def T_int_numerical(self, T_i, T_w):
        """Numerical evaluation of T_int (as opposed to the symbolic one)."""
        rho = self.rho_w.value
        if self.is_layered:
            ri = np.array(self.r_i.x.array)
            rw = np.array(self.r_w.x.array)
        else:
            ri = self.r_i.value
            rw = self.r_w.value
        weights = [self.K_i.value/ri,
                   self.K_w.value/rw,
                   rho*self.L_sol.value*self.R_m.value]
        numerator = (weights[0]*np.array(T_i.x.array)
                     + weights[1]*np.array(T_w.x.array)
                     + weights[2]*self.T_melt.value)
        denominator = (weights[0] + weights[1] + weights[2])
        return numerator/denominator

    def W_SSA_numerical(self, Se, phi):
        """Numerical evaluation of the wet specific surface area."""
        part1 = ((Se )*np.array(phi.x.array) *
                 np.log(np.array(phi.x.array)))
        if self.is_layered:
            phi0 = 1 - np.array(self.rho_s.x.array)/self.rho_i.value
        else:
            phi0 = 1 - self.rho_s.value/self.rho_i.value
        return part1*self.SSA_0.value / (phi0*np.log(phi0))

    def make_into_dict(self):
        """Store attributes into dictionnary."""
        d = {}
        p_layers = ["d_i", "rho_s", "r_i", "r_w", "alpha", "N"]
        for key, value in vars(self).items():
            if self.is_layered and key in p_layers:
                d[key] = value.x.array
            elif (key == "layer_params_dict"
                  and self.layer_params_dict is not None):
                for k, val in value.items():
                    funcString = str(inspect.getsourcelines(val["locator"])[0])
                    funcString = funcString.strip("['\\n']").split(" = ")[0]
                    val["locator"] = funcString
                d[key] = value
            elif key != "is_layered" and key != "layer_params_dict":
                d[key] = float(value.value)
            elif key != "layer_params_dict":
                d[key] = value
        return d
