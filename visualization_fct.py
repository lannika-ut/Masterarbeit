import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
import pyvista as pv # type: ignore
from dolfinx import fem, io, mesh, plot, geometry # type: ignore


def plot_mesh(V, title="Mesh for finite element method"):
    pv.set_jupyter_backend("html") # "static" should work every time
    cells, types, x = plot.vtk_mesh(V) # convert mesh to vtk data which pyvista can read
    grid = pv.UnstructuredGrid(cells, types, x)
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.show_axes()
    plotter.add_title(title)    
    if not pv.OFF_SCREEN:
        plotter.show()
    else:
        print("pyvista needs to be used in the default setting of pyvista.OFF_SCREEN=False.")
 
def plot_scalar_function(V, u, warped=False, name = "u", title="", fct_as_array=False, cmap = "viridis"):
    """
    Plot a dolfinx.function on its grid.

    Args:
        V (dolfinx.fem.function.FunctionSpace): Functionspace of the function, containing the grid.
        u (dolfinx.fem.function.Function): Scalar function that should be plotted.
        warped (bool, optional): If the plot should be warped to see changes in function values in 3D or not. Defaults to False.
        name (String, optional): Name of the function to add to colour bar.
        title (String, optional): Title of the plot.
        fct_as_array (bool, optional): If the function to be plotted is already in form of a numpy array. Defaults to False.
        cmap (String, optional): Colour map to use. Defaults to viridis.
    """
    pv.set_jupyter_backend("static")
    grid = pv.UnstructuredGrid(*plot.vtk_mesh(V))
    if fct_as_array:
        grid.point_data[name] = u
    else:
        grid.point_data[name] = u.x.array.real
    grid.set_active_scalars(name)
    plotter = pv.Plotter()
    if warped:
        warp = grid.warp_by_scalar()
        plotter.add_mesh(warp, show_edges = True, show_scalar_bar = True, cmap = cmap)
    else:
        plotter.add_mesh(grid, show_edges = True, cmap = cmap)
    plotter.view_xy()
    plotter.add_axes()
    plotter.show_axes()
    a = plotter.add_title(title)
    if not pv.OFF_SCREEN:
        plotter.show()
    else:
        print("pyvista needs to be used in the default setting of pyvista.OFF_SCREEN=False.")

def eval_fct_on_grid(grid, u, domain):
    """
    Evaluate a scalar or vector-valued dolfinx function on a grid.

    Args:
        grid (np.ndarray): List of 2D points of the grid, size (nx*nz, 2).
        u (dolfinx.fem.Function): Scalar or vector-valued function to be evaluated.
        domain (dolfinx.mesh): Domain on which the function is defined.

    Raises:
        RuntimeError: If the points are not on the processor.

    Returns:
        np.ndarray: Evaluated function. If u was scalar, the size is (nx*nz,), if u was vector-valued of dimension k, the size is (nx*nz, k).
    """

    points = np.column_stack(
        (grid[:, 0], grid[:, 1], np.zeros(len(grid)))
    ).astype(np.float64)

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(
        domain, cell_candidates, points
    )

    cells = []
    points_on_proc = []
    point_ids = []

    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            cells.append(links[0])
            points_on_proc.append(points[i])
            point_ids.append(i)

    if len(points_on_proc) == 0:
        raise RuntimeError("No points found on this MPI rank.")

    evaluated = u.eval(
        np.asarray(points_on_proc, dtype=np.float64),
        np.asarray(cells, dtype=np.int32),
    )
    
    evaluated = np.asarray(evaluated)

    # Scalar field
    if evaluated.ndim == 1 or evaluated.shape[1] == 1:
        values = np.full(len(points), np.nan)
        values[point_ids] = evaluated.ravel()

    # Vector/tensor field
    else:
        values = np.full((len(points), evaluated.shape[1]), np.nan)
        values[point_ids] = evaluated

    return values

def get_grid(P0, P1, P2, P3, nx, nz):
    """
    Generate a grid of points made up of four corner points.

    Args:
        P0 (np.array): x and y coordinates of the bottom-left corner point
        P1 (np.array): x and y coordinates of the bottom-right corner point
        P2 (np.array): x and y coordinates of the top-right corner point
        P3 (np.array): x and y coordinates of the top-left corner point
        nx (int): number of points in x-direction.
        nz (int): number of points in z-direction.

    Returns:
        np.ndarray: list of 2D points of the grid, size (nx*nz, 2)
        np.ndarray: plotting points for x-coordinates, size (nx, nz)
        np.ndarray: plotting points for z-coordinates, size (nx, nz)
    """
    x_int = np.linspace(0, 1, nx)
    z_int = np.linspace(0, 1, nz)
    x_int, z_int = np.meshgrid(x_int, z_int)
    # Bilinear interpolation
    x_grid = (1 - x_int) * (1 - z_int) * P0[0] + x_int * (1 - z_int) * P1[0] + x_int * z_int * P2[0] + (1 - x_int) * z_int * P3[0]
    z_grid = (1 - x_int) * (1 - z_int) * P0[1] + x_int * (1 - z_int) * P1[1] + x_int * z_int * P2[1] + (1 - x_int) * z_int * P3[1]
    # Combine into a grid of points
    grid = np.column_stack((x_grid.ravel(), z_grid.ravel()))
    return grid, x_grid, z_grid

def view_boundaries(domain, facet_tags, tag):
    # < visualise dofs of facets with pvista >

    V = fem.FunctionSpace(domain, ("CG", 1))
    grid = pv.UnstructuredGrid(*plot.vtk_mesh(domain))

    # Locate degrees of freedom associated with the specific tag
    facet_indices = facet_tags.indices[facet_tags.values == tag]
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facet_indices)

    # return coords of highlighted nodes:
    coords = V.tabulate_dof_coordinates()
    dof_coords = coords[dofs]

    # highlight dof coordinates:
    points = pv.PolyData(dof_coords)
    points["dofs"] = np.zeros(dof_coords.shape[0]) 

    plotter = pv.Plotter()
    plotter.add_mesh(grid, color="lightgray", show_edges=True)  # Show the mesh
    plotter.add_points(points, color="red", point_size=10) # Show highlighted points
    plotter.camera_position = [(7, 6, 1), (7, 6, 0), (0, 0, 0)] 
    plotter.camera.zoom(0.18)  
    plotter.show()