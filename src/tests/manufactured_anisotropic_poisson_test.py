import unittest

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import scipy.sparse as sparse
import path_magic
from assemble import assemble
from basis_forms import BasisForm
from coboundaries import d
from forms import Form
from function_space import FunctionSpace
from inner_product import MeshFunction, inner
from mesh import CrazyMesh


class TestAnisotropicPoisson(unittest.TestCase):

    def solution(self, x, y):
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    @unittest.skip
    def test_single_element(self):
        """Test the anysotropic case in a single element."""
        dim = 2
        elements_layout = (1, 1)
        bounds_domain = ((-1, 1), (-1, 1))
        curvature = 0.1

        p = (20, 20)
        is_inner = False

        crazy_mesh = CrazyMesh(dim, elements_layout, bounds_domain, curvature)

        func_space_2_lobatto = FunctionSpace(crazy_mesh, '2-lobatto', p, is_inner)
        func_space_1_lobatto = FunctionSpace(crazy_mesh, '1-lobatto', p, is_inner)

        def diffusion_11(x, y): return 4 * np.ones(np.shape(x))

        def diffusion_12(x, y): return 3 * np.ones(np.shape(x))

        def diffusion_22(x, y): return 5 * np.ones(np.shape(x))

        mesh_k = MeshFunction(crazy_mesh)
        mesh_k.continous_tensor = [diffusion_11, diffusion_12, diffusion_22]

        basis_1 = BasisForm(func_space_1_lobatto)
        basis_1.quad_grid = 'gauss'
        basis_2 = BasisForm(func_space_2_lobatto)
        basis_2.quad_grid = 'gauss'

        phi_2 = Form(func_space_2_lobatto)
        phi_2.basis.quad_grid = 'gauss'

        M_1k = inner(basis_1, basis_1, mesh_k)

        N_2 = inner(basis_2, d(basis_1))

        def source(x, y):
            return -36 * np.pi ** 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + 24 * np.pi ** 2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

        form_source = Form(func_space_2_lobatto)
        form_source.discretize(source)

        M_2 = inner(basis_2, basis_2)

        lhs = np.vstack((np.hstack((M_1k[:, :, 0], N_2[:, :, 0])), np.hstack(
            (np.transpose(N_2[:, :, 0]), np.zeros((np.shape(N_2)[1], np.shape(N_2)[1]))))))

        rhs = np.zeros((np.shape(lhs)[0], 1))
        rhs[-np.shape(M_2)[0]:] = form_source.cochain @ M_2

        phi_2.cochain = np.linalg.solve(lhs, rhs)[-phi_2.basis.num_basis:].flatten()
        xi = eta = np.linspace(-1, 1, 200)
        phi_2.reconstruct(xi, eta)
        (x, y), data = phi_2.export_to_plot()
        print("max value {0} \nmin value {1}" .format(np.max(data), np.min(data)))
        npt.assert_array_almost_equal(self.solution(x, y), data, decimal=2)

    def test_multiple_elements(self):
        """Test the anysotropic case in multiple element."""
        p = (6, 6)
        is_inner = False
        crazy_mesh = CrazyMesh(2, (8, 5),  ((-1, 1), (-1, 1)), curvature=0.1)

        func_space_2_lobatto = FunctionSpace(crazy_mesh, '2-lobatto', p, is_inner)
        func_space_1_lobatto = FunctionSpace(crazy_mesh, '1-lobatto', p, is_inner)
        func_space_1_lobatto.dof_map.continous_dof = True

        def diffusion_11(x, y): return 4 * np.ones(np.shape(x))

        def diffusion_12(x, y): return 3 * np.ones(np.shape(x))

        def diffusion_22(x, y): return 5 * np.ones(np.shape(x))

        def source(x, y):
            return -36 * np.pi ** 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + 24 * np.pi ** 2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

        # mesh function to inject the anisotropic tensor
        mesh_k = MeshFunction(crazy_mesh)
        mesh_k.continous_tensor = [diffusion_11, diffusion_12, diffusion_22]

        # definition of the basis functions
        basis_1 = BasisForm(func_space_1_lobatto)
        basis_1.quad_grid = 'gauss'
        basis_2 = BasisForm(func_space_2_lobatto)
        basis_2.quad_grid = 'gauss'

        # solution form
        phi_2 = Form(func_space_2_lobatto)
        phi_2.basis.quad_grid = 'gauss'

        form_source = Form(func_space_2_lobatto)
        form_source.discretize(source, ('gauss', 30))
        # find inner product
        M_1k = inner(basis_1, basis_1, mesh_k)
        N_2 = inner(d(basis_1), basis_2)
        M_2 = inner(basis_2, basis_2)
        # assemble
        M_1k = assemble(M_1k, func_space_1_lobatto)
        N_2 = assemble(N_2, (func_space_1_lobatto, func_space_2_lobatto))
        M_2 = assemble(M_2, func_space_2_lobatto)

        lhs = sparse.bmat([[M_1k, N_2], [N_2.transpose(), None]]).tocsc()
        rhs_source = (form_source.cochain @ M_2)[:, np.newaxis]
        rhs_zeros = np.zeros(lhs.shape[0] - np.size(rhs_source))[:, np.newaxis]
        rhs = np.vstack((rhs_zeros, rhs_source))

        solution = sparse.linalg.spsolve(lhs, rhs)
        phi_2.cochain = solution[-func_space_2_lobatto.num_dof:]

        # sample the solution
        xi = eta = np.linspace(-1, 1, 200)
        phi_2.reconstruct(xi, eta)
        (x, y), data = phi_2.export_to_plot()
        plt.contourf(x, y, data)
        plt.show()
        print("max value {0} \nmin value {1}" .format(np.max(data), np.min(data)))
        npt.assert_array_almost_equal(self.solution(x, y), data, decimal=2)


if __name__ == '__main__':
    unittest.main()
