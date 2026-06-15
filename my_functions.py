import numpy as np
from dolfinx import mesh, fem # type: ignore
import ufl

def create_quad_domain(comm, nx, ny, p0, p1, p2, p3, celltype=mesh.CellType.triangle):
    """
    Create a domain defined by four corner points.

    Args:
        comm (mpi4py.MPI.Intracomm): MPI communicator
        nx (int): number of cells in x-direction
        ny (int): number of cells in y-direction
        p0 (np.array): x and y coordinates of the bottom-left corner point
        p1 (np.array): x and y coordinates of the bottom-right corner point
        p2 (np.array): x and y coordinates of the top-right corner point
        p3 (np.array): x and y coordinates of the top-left corner point

    Returns:
        dolfinx.mesh: Domain defined by the corner points.
    """

    msh = mesh.create_unit_square(comm, nx, ny, cell_type=celltype)

    x = msh.geometry.x
    xi = x[:, 0]
    eta = x[:, 1]

    x[:, :2] = (
        np.outer((1-xi)*(1-eta), p0) +
        np.outer(xi*(1-eta), p1) +
        np.outer(xi*eta, p2) +
        np.outer((1-xi)*eta, p3)
    )
    return msh

def print_matrix_from_equation(Eq):
    X = fem.petsc.assemble_matrix(fem.form(Eq))
    X.assemble()
    X.convert("dense")
    C = X.getDenseArray()
    print(f"Matrix form: {C}")

def assign_material(domain, Q, layer_params):
    """
    Assign material properties to functions to account for different layer properties.

    Args:
        domain (dolfinx.mesh.Mesh): FEM domain
        Q (dolfinx.fem.function.FunctionSpace): Functionspace for parameter functions, most likely DG0
        layer_params (dict): Dictionnary containing the van Genuchten parameters alpha, N and the soil parameters Ks, theta_r, theta_s. It also contains a boolean function to locate the layer within the domain named locator.
    Returns:
        dict: dictionnary of the parameter functions.
    """
    # create parameter functions
    Ks = fem.Function(Q)
    alpha = fem.Function(Q)
    N = fem.Function(Q)
    theta_r = fem.Function(Q)
    theta_s = fem.Function(Q)

    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)

    midpoints = mesh.compute_midpoints(domain, tdim, cells)

    Ks_vals = np.zeros(num_cells)
    alpha_vals = np.zeros(num_cells)
    n_vals = np.zeros(num_cells)
    theta_r_vals = np.zeros(num_cells)
    theta_s_vals = np.zeros(num_cells)

    for c, x in enumerate(midpoints):
        # Check in which layer the midpoint is and assign the corresponding parameters
        for key, value in layer_params.items():
            if value["locator"](x):
                Ks_vals[c] = value["Ks"]
                alpha_vals[c] = value["alpha"]
                n_vals[c] = value["N"]
                theta_r_vals[c] = value["theta_r"]
                theta_s_vals[c] = value["theta_s"]
    # Fill functions with right values
    Ks.x.array[:] = Ks_vals
    alpha.x.array[:] = alpha_vals
    N.x.array[:] = n_vals
    theta_r.x.array[:] = theta_r_vals
    theta_s.x.array[:] = theta_s_vals

    Ks.x.scatter_forward()
    alpha.x.scatter_forward()
    N.x.scatter_forward()
    theta_r.x.scatter_forward()
    theta_s.x.scatter_forward()
    # arange into dict
    _dict = {"Ks": Ks, 
            "alpha": alpha,
            "N": N,
            "theta_r": theta_r, 
            "theta_s": theta_s}
    return _dict

# van Genuchten parametrizations
def S_e(h_w, alpha, N):
    return ufl.conditional(h_w < 0, (1 + (- alpha * h_w)**N)**((1 - N) / N), 1)
def theta(Se, theta_r, theta_s):
    return theta_r + (theta_s - theta_r)*Se
def k_rel(Se, N):
    m = 1 - 1/N
    return ufl.conditional(Se < 1 - 1e-7, ufl.sqrt(Se) * (1 - (1 - Se**(1/m))**m)**2 , 1)