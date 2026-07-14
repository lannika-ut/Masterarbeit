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
        self.S_r = fem.Constant(domain, PETSc.ScalarType(1e-6)) # residual saturation
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
            self.min_hw = param_fct["min_hw"]
            self.layer_params_dict = layer_params
        else:  # homogeneous snow
            self.is_layered = False
            self.d_i = fem.Constant(domain, PETSc.ScalarType(2e-3))  # m
            self.r_i = fem.Constant(domain,
                                    PETSc.ScalarType(0.06*self.d_i.value))  # m
            self.r_w = fem.Constant(domain,
                                    PETSc.ScalarType(1.35*self.d_i.value))  # m
            self.rho_s = fem.Constant(domain, PETSc.ScalarType(350))  # kg/m^3
            a = (4.4e6) * (self.rho_s.value/self.d_i.value)**(-0.98)  # 1/m
            n = 1 + (2.7e-3) * (self.rho_s.value/self.d_i.value)**(0.61)
            self.alpha = fem.Constant(domain, PETSc.ScalarType(a))
            self.N = fem.Constant(domain, PETSc.ScalarType(n))
            self.min_hw = fem.Constant(
                domain, PETSc.ScalarType(self.calc_min_hw()))

        self.theta_r = fem.Constant(domain, PETSc.ScalarType(0.02))
        self.SSA_0 = fem.Constant(domain, PETSc.ScalarType(4114))  # 1/m

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
        minhw = fem.Function(Q)

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
        minhw_vals = np.zeros(num_cells)

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
                    minhw_vals[c] = (-(self.S_r.value**(n_vals[c]/(1-n_vals[c]))
                                       - 1)**(1/n_vals[c])/alpha_vals[c])

        # Fill functions with right values
        alpha.x.array[:] = alpha_vals
        N.x.array[:] = n_vals
        d_i.x.array[:] = d_i_vals
        rho_s.x.array[:] = rho_s_vals
        r_i.x.array[:] = ri_vals
        r_w.x.array[:] = rw_vals
        minhw.x.array[:] = minhw_vals

        alpha.x.scatter_forward()
        N.x.scatter_forward()
        d_i.x.scatter_forward()
        rho_s.x.scatter_forward()
        r_i.x.scatter_forward()
        r_w.x.scatter_forward()
        minhw.x.scatter_forward()
        # arange into dict
        _dict = {"d_i": d_i,
                 "rho_s": rho_s,
                 "alpha": alpha,
                 "N": N,
                 "r_i": r_i,
                 "r_w": r_w,
                 "min_hw": minhw,
                 }
        return _dict

    def calc_min_hw(self):
        """Calculate min. pressure head such that saturation stays above 0.001."""
        minhw = (
            -(self.S_r.value**(self.N.value/(1-self.N.value))
              - 1)**(1/self.N.value)/self.alpha.value)
        return minhw

    def S_e(self, h_w):
        """Calculate the effective saturation after van Genuchten."""
        if self.is_layered:
            h_w_safe = ufl.max_value(h_w, self.min_hw)
        else:
            h_w_safe = ufl.max_value(h_w, self.min_hw)
        h_w_safe = ufl.min_value(h_w_safe, -1e-8)
        return ufl.conditional(
            h_w < -1e-8,
            (1 + (-self.alpha*h_w_safe)**self.N) ** ((1-self.N)/self.N),
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
        part1 = (Se - self.S_r) * ufl.ln(phi) * phi
        phi0 = 1 - self.rho_s/self.rho_i
        return part1*self.SSA_0 / (phi0*ufl.ln(phi0))

    def S_e_numerical(self, h_w):
        """Numerical evaluation of the effective saturation after van Genuchten."""
        hw = np.array(h_w.x.array)
        Se = np.ones_like(hw)
        if self.is_layered:
            a = np.array(self.alpha.x.array)
            n = np.array(self.N.x.array)
            hw_safe = np.clip(hw, self.min_hw.x.array, -1e-8)
            Se[hw < -1e-8] = ((1 + (-a[hw < -1e-8]*hw_safe[hw < -1e-8])
                               ** n[hw < -1e-8])
                               ** ((1-n[hw < -1e-8])/n[hw < -1e-8]))
        else:
            a = self.alpha.value
            n = self.N.value
            hw_safe = np.clip(hw, self.min_hw.value, -1e-8)
            Se[hw < -1e-8] = (1 + (-a*hw_safe[hw < -1e-8])**n) ** ((1-n)/n)
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
        part1 = ((Se-self.S_r.value)*np.array(phi.x.array) *
                 np.log(np.array(phi.x.array)))
        if self.is_layered:
            phi0 = 1 - np.array(self.rho_s.x.array)/self.rho_i.value
        else:
            phi0 = 1 - self.rho_s.value/self.rho_i.value
        return part1*self.SSA_0.value / (phi0*np.log(phi0))

    def calc_krel(self, hw, alpha, N, minhw):
        """Calculate the relative permeability for a given pressure head.

        Args:
            hw (float): pressure head.
            alpha (float): value of the van Genuchten parameter alpha.
            N (float): value of the van Genuchten parameter N.
        Returns:
            float: value of the relative permeability.
        """
        m = 1 - 1/N
        Se = 1
        krel = 1
        if hw < -1e-8:
            hw = np.clip(hw, minhw, None)
            Se = (1 + (-alpha*hw)**N) ** ((1-N)/N)
            if Se < 1-1e-7:
                krel = np.sqrt(Se) * (1 - (1-Se**(1/m))**m)**2
        return krel

    def upwind_krel(self, hw, domain):
        """Calculate the relative permeability in an upwind scheme, i.e. for each cell, take the h_w value of the cell's node where h_tot is the highest for calculating k_rel.

        Args:
            hw (dolfinx.fem.Function): Pressure head function.
            domain (dolfinx.mesh.Mesh): The domain.

        Returns:
            dolfinx.fem.Function: DG0 function containing the upwind k_rel values.
        """
        Q = fem.functionspace(domain, ("DG", 0))
        krel = fem.Function(Q)
        # Find out which nodes belong to which element
        tdim = domain.topology.dim
        c2v = domain.topology.connectivity(tdim, 0)  # element -> nodes
        num_cells = domain.topology.index_map(tdim).size_local
        cells = np.arange(num_cells, dtype=np.int32)
        for cell in cells:
            node_index = c2v.links(cell)
            node_coords = domain.geometry.x[node_index]
            h_tot = hw.x.array[node_index] + node_coords[:, 1]  # h_tot = hw + z
            max_hw = hw.x.array[node_index[np.argmax(h_tot)]]
            if self.is_layered:
                alpha = self.alpha.x.array[cell]
                N = self.N.x.array[cell]
                minhw = self.min_hw.x.array[cell]
            else:
                alpha = self.alpha.value
                N = self.N.value
                minhw = self.min_hw.value
            krel.x.array[cell] = self.calc_krel(max_hw, alpha, N, minhw)
            krel.x.scatter_forward()
        return krel

    def calc_source_term(self, hw, phi, Ti, Tw):
        """Calculate the source term of the mass conservation laws as a DG0 function.

        Args:
            hw (fem.Function): Pressure head.
            phi (fem.Function): Porosity in a DG0 function.
            Ti (fem.Function): Ice temperature.
            Tw (fem.Function): Water temperature.

        Returns:
            fem.Function: R_m*W_SSA*(T_int - T_melt)
        """
        Q = phi.function_space
        src = fem.Function(Q)
        hw_dg0 = fem.Function(Q)
        hw_dg0.interpolate(hw)
        Ti_dg0 = fem.Function(Q)
        Ti_dg0.interpolate(Ti)
        Tw_dg0 = fem.Function(Q)
        Tw_dg0.interpolate(Tw)
        Wssa = self.W_SSA_numerical(self.S_e_numerical(hw_dg0), phi)
        Tint = self.T_int_numerical(Ti_dg0, Tw_dg0)
        src.x.array[:] = self.R_m.value*Wssa*(Tint-self.T_melt.value)
        return src

    def make_into_dict(self):
        """Store attributes into dictionnary."""
        d = {}
        p_layers = ["d_i", "rho_s", "r_i", "r_w", "alpha", "N", "min_hw"]
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
