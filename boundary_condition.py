import numpy as np
from dolfinx import fem, mesh
import ufl
from petsc4py import PETSc


class BoundaryCondition:
    """Class for creating Dirichlet and Neumann boundary conditions on the same finite element mesh but on different functionspaces.
    """

    def __init__(self, domain, boundaries):
        """Initiate, create custom integration measure on finite element domain.

        Args:
            domain (Dolfinx.mesh): Finite element mesh.
            boundaries (dict): Dictionnary containing the boundary marker (integer) as key and a boolean locator function as value for each boundary. {marker: lambda x: True_if_on_boundary}
        """
        self.boundaries = boundaries
        self.domain = domain
        self.ds = self._create_boundary_integration_measure()

    def _create_boundary_integration_measure(self):
        """Create a custom integration measure for each boundary.

        Args:
            domain (Dolfinx.mesh): Finite element mesh.

        Returns:
            ufl.Measure: Integration measure customised such that ds(marker) integrates over the boundary facets located by the corresponding locator.
        """
        facet_indices, facet_markers = [], []
        fdim = self.domain.topology.dim - 1
        for marker, locator in self.boundaries.items():
            facets = mesh.locate_entities(self.domain, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = mesh.meshtags(
            self.domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
        )
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=facet_tag)
        return ds

    def make_boundary_condition(self, bc_dict):
        """Create dolfinx boundary conditions for a set of given boundary conditions for different functionspaces.

        Args:
            bc_dict (dict): Dictionnary containing all relevant information. No-flow boundary conditions do not have to be specified and are used for every boundary that is not explicitely given. 
                bc_dict = {
                key: {
                    "marker": marks_boundary (int),
                    "name": type_of_bc("Dirichlet" or "Neumann"),
                    "value": boundary_value_or_function,
                    "functionspace": fem.functionspace,
                    "testfunction": ufl.TestFunction
                    },
                    ...
                }.

        Raises:
            TypeError: Boundary condition unknown (name neither Dirichlet nor Neumann).

        Returns:
            dict: Dictionnary containing either the fem.dirichletbc or the integral over the Neumann boundary of the function*testfunction. {key: dirichlet_or_Neumann_bc}. The key is the same as in bc_dict.
        """
        bcs = {}
        for key, values in bc_dict.items():
            marker = values["marker"]
            if values["name"] == "Dirichlet":
                V = values["functionspace"]
                u_D = fem.Function(V)
                if callable(values["value"]):
                    u_D.interpolate(values["value"])
                else:
                    u_D.x.array[:] = fem.Constant(
                        self.domain, PETSc.ScalarType(values["value"]))
                dofsD = fem.locate_dofs_geometrical(V, self.boundaries[marker])
                bc = fem.dirichletbc(u_D, dofsD)
            elif values["name"] == "Neumann":
                bc = values["testfunction"]*values["value"]*self.ds(marker)
            else:
                raise ValueError(
                    f"Unknown boundary condition, maybe you misspelled. Accepted are 'Dirichlet' and 'Neumann'. Got: {values['name']}")
            bcs[key] = bc
        return bcs
