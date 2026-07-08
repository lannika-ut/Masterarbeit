from dolfinx import fem
from petsc4py import PETSc
from dolfinx.fem.petsc import (
    assemble_matrix,
    apply_lifting,
    assemble_vector,
    set_bc,
    )
import ufl

class NonlinearPDE_SNESProblem:
    """Nonlinear problem class for a PDE problem using SNES interface."""

    def __init__(self, F, u, bc):
        """Initialize nonlinear PDE problem.

        Args:
            F (ufl problem): Residual of the problem. 
            u (dolfinx.fem.Function): Trial function in form of a fem.Function of the problem.
            bc (dolfinx dirichletbc): Dirichlet boundary conditions.
        """
        
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.form(F)
        self.a = fem.form(ufl.derivative(F, u, du))
        self.bc = bc
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                       mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        if self.bc is not None:
            apply_lifting(F, [self.a], bcs=[self.bc], x0=[x], alpha=-1.0)
            F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(F, self.bc, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                       mode=PETSc.ScatterMode.FORWARD)
        J.zeroEntries()
        if self.bc is not None:
            assemble_matrix(J, self.a, bcs=self.bc)
        else:
            assemble_matrix(J, self.a)
        J.assemble()