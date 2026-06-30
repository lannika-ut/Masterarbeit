import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
import pyvista as pv # type: ignore
from dolfinx import fem, io, mesh, plot, geometry # type: ignore


def plot_mesh(V, title="Mesh for finite element method"):
    pv.set_jupyter_backend("html")
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

def plot_mesh2(mesh: mesh.Mesh, values=None, title="Mesh for finite element method"):
    """
    Given a DOLFINx mesh, create a `pyvista.UnstructuredGrid`,
    and plot it and the mesh nodes.

    Args:
        mesh: The mesh we want to visualize
        values: List of values indicating a marker for each cell in the mesh

    Note:
        If `values` are given as input, they are assumed to be a marker
        for each cell in the domain.
    """
    pv.set_jupyter_backend("static")
    # We create a pyvista plotter instance
    plotter = pv.Plotter()

    # Since the meshes might be created with higher order elements,
    # we start by creating a linearized mesh for nicely inspecting the triangulation.
    V_linear = fem.functionspace(mesh, ("Lagrange", 1))
    linear_grid = pv.UnstructuredGrid(*plot.vtk_mesh(V_linear))

    # If the mesh is higher order, we plot the nodes on the exterior boundaries,
    # as well as the mesh itself (with filled in cell markers)
    if mesh.geometry.cmap.degree > 1:
        ugrid = pv.UnstructuredGrid(*plot.vtk_mesh(mesh))
        if values is not None:
            ugrid.cell_data["Marker"] = values
        plotter.add_mesh(ugrid, style="points", color="b", point_size=10)
        ugrid = ugrid.tessellate()
        plotter.add_mesh(ugrid, show_edges=False)
        plotter.add_mesh(linear_grid, style="wireframe", color="black")
    else:
        # If the mesh is linear we add in the cell markers
        if values is not None:
            linear_grid.cell_data["Marker"] = values
        plotter.add_mesh(linear_grid, show_edges=True)

    # We plot the coordinate axis and align it with the xy-plane
    plotter.show_axes()
    plotter.add_title(title)   
    plotter.view_xy()
    if not pv.OFF_SCREEN:
        plotter.show()


def evaluate_fct(domain, points, fct):
    """
    This is a wrapper function to evaluate multiple dolfinx.fem.function.Function at given points. Explanations to what is going on can either be found in the script FEM_DeflectionOfAMembrane.ipynb or online: https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html.

    Args:
        domain (dolfinx.mesh.Mesh): Mesh containing the topology (i.e. the cells).
        points (np.ndarray): Points at which the functions should be evaluated. Shape should be (3, num_points) with x-coordinates in the first, y-coordinates in the second and z-coordinates in the third dimension.
        fct (dolfinx.fem.function.Function): Function that needs to be evaluated. The function should be a linear combination of the basis functions on the domain, either created by interpolating an expression on a functionspace or by a finite element algorithm.

    Returns:
        fcts_values (np.ndarray): evaluated points and nan for points outside the mesh or the process.
    """
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding box collide with the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, 
                                                       cell_candidates, 
                                                       points.T)
    
    points_not_on_proc = []
    fct_values = []
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0 and len(points_not_on_proc) == 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
        if len(colliding_cells.links(i)) > 0 and len(points_not_on_proc) > 0:
            points_on_proc.append(point)
            cell_link = colliding_cells.links(i)[0]
            cells.append(cell_link)
            fct_values.append(fct.eval(point, cell_link))
        else:
            # First time a point is not on the processor: evaluate all valid points
            if len(points_on_proc) == 0:
                fct_values.append(fct.eval(np.array(points_on_proc, dtype=np.float64), cells))
            points_not_on_proc.append(point)
            fct_values.append(np.nan)
    fct_values = np.hstack(fct_values)
    return fct_values
        

def plotScalarFunction(V, u, warped=False, name = "u", title="", fct_as_array=False, cmap = "viridis"):
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
    Evaluate a dolfinx.function u on a grid.

    Args:
        grid (np.ndarray, shape (x,2)): grid containing values in x-direction in grid[:,0] and values in z-direction in grid[:,1]
        u (dolfinx.fem.function.Function): scalar function
        domain (dolfinx.mesh.Mesh): mesh containing the topology
    Returns:
        np.array (length of grid[:,0]): function values of u on the grid.
    """
    p = np.vstack([grid[:,0], grid[:,1], np.zeros(len(grid[:,0]))])
    values = evaluate_fct(domain, p, u)
    h = np.array(values).flatten()
    return h

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

    # Generate plotting points
    x_int = np.linspace(0, 1, nx+1)
    z_int = np.linspace(0, 1, nz+1)
    x_int, z_int = np.meshgrid(x_int, z_int)
    # Bilinear interpolation
    x_plot = (1 - x_int) * (1 - z_int) * P0[0] + x_int * (1 - z_int) * P1[0] + x_int * z_int * P2[0] + (1 - x_int) * z_int * P3[0]
    z_plot = (1 - x_int) * (1 - z_int) * P0[1] + x_int * (1 - z_int) * P1[1] + x_int * z_int * P2[1] + (1 - x_int) * z_int * P3[1]

    return grid, x_grid, z_grid