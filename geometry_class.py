import numpy as np
from mpi4py import MPI
from dolfinx import mesh


class Geometry:
    """Class for creating the geometry and the finite element domain."""

    def __init__(
            self, height=1, length=1, slope=0, bottom_left=np.array([0, 0])):
        """Create the geometry.

        Args:
            height (float, optional): Height of the 2D-domain. Defaults to 1.
            length (float, optional): Length of the 2D-domain. Defaults to 1.
            bottom_left (np.array, optional): x- and y-coordinates of the bottom left corner point of the domain. Defaults to np.array([0,0]).
        """
        self.height = height
        self.length = length
        self.slope = slope
        self.corner_points = self._calculate_corner_points(bottom_left)
        self.nx = 0
        self.nz = 0

    def _calculate_corner_points(self, bottom_left):
        """Calculate the corner points of the domain based on height, length and slope.

        Args:
            bottom_left (np.array): x- and y-coordinates of the bottom left corner point of the domain.

        Returns:
            list: List of corner point coordinates [bottom_left, bottom_right, top_right, top_left].
        """
        x0 = bottom_left[0]
        y0 = bottom_left[1]
        bottom_right = np.array(
            [x0 + self.length, y0 + self.slope*self.length])
        top_right = np.array([bottom_right[0], bottom_right[1] + self.height])
        top_left = np.array([x0, y0 + self.height])
        return [bottom_left, bottom_right, top_right, top_left]

    def make_domain(
            self, nx=20, nz=20, comm=MPI.COMM_WORLD,
            celltype=mesh.CellType.quadrilateral):
        """Make a finite element mesh based on the geometry.

        Args:
            nx (int, optional): Number of cells in x-direction. Defaults to 20.
            nz (int, optional): Number of cells in y-direction. Defaults to 20.
            comm (MPI Communicator, optional): The MPI Communicator. Defaults to MPI.COMM_WORLD.
            celltype (Dolfinx.mesh.Celltype, optional): The preferred celltype (triangle or quadrilateral). Defaults to mesh.CellType.quadrilateral.

        Returns:
            Dolfinx.mesh: Finite element mesh.
        """
        self.nx = nx
        self.nz = nz
        domain = mesh.create_unit_square(
            comm, self.nx, self.nz, cell_type=celltype)
        x = domain.geometry.x
        xi = x[:, 0]
        eta = x[:, 1]
        x[:, :2] = (
            np.outer((1-xi)*(1-eta), self.corner_points[0]) +
            np.outer(xi*(1-eta), self.corner_points[1]) +
            np.outer(xi*eta, self.corner_points[2]) +
            np.outer((1-xi)*eta, self.corner_points[3])
        )
        return domain

    def make_into_dict(self):
        """Store attributes into dictionnary."""
        d = {}
        for key, value in vars(self).items():
            d[key] = value
        return d
    
    def make_from_dict(self, d):
        """Set attributes from dictionnary d (output of make_into_dict())."""
        for key, value in d.items():
            setattr(self, key, value)