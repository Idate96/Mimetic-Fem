import os
import path_magic
import unittest
from exceptions import QuadratureError
from inner_product import MeshFunction, diff_tens_11, diff_tens_12, diff_tens_22
import numpy as np
import numpy.testing as npt
from basis_forms import BasisForm
from function_space import FunctionSpace
from inner_product import inner
from mesh import CrazyMesh


class TestBasis1Form(unittest.TestCase):
    # @unittest.skip
    def test_exceptions(self):
        p = 2
        mesh = CrazyMesh(2, (2, 2), ((-1, 1), (-1, 1)))
        func_space = FunctionSpace(mesh, '1-lobatto', p)
        basis = BasisForm(func_space)
        basis_1 = BasisForm(func_space)
        quad_cases = [(None, 'gauss'), (None, None), ('gauss', None), ('lobatto', 'gauss')]
        for case in quad_cases:
            if None in case:
                with self.assertRaises(UnboundLocalError):
                    basis.quad_grid = case[0]
                    basis_1.quad_grid = case[1]
            else:
                basis.quad_grid = case[0]
                basis_1.quad_grid = case[1]
                self.assertRaises(QuadratureError, inner, basis, basis_1)

    # @unittest.skip
    def test_inner(self):
        """Test inner product of one forms."""
        list_cases = ['p2_n2-2', 'p2_n3-2',
                      'p5_n1-10', 'p10_n2-2', 'p13_n12-8']
        p = [2, 2, 5, 10, 13]
        n = [(2, 2), (3, 2), (1, 10), (2, 2), (12, 8)]
        curvature = [0.1, 0.1, 0.1, 0.1, 0.1]
        for i, case in enumerate(list_cases[:-1]):
            M_1_ref = np.loadtxt(
                os.getcwd() + '/src/tests/test_M_1/M_1k_' + case + '.dat', delimiter=',').reshape(2 * p[i] * (p[i] + 1),  n[i][0] * n[i][1], 2 * p[i] * (p[i] + 1))
            my_mesh = CrazyMesh(
                2,  n[i], ((-1, 1), (-1, 1)), curvature=curvature[i])
            function_space = FunctionSpace(my_mesh, '1-lobatto', p[i])
            form = BasisForm(function_space)
            form.quad_grid = 'gauss'
            form_1 = BasisForm(function_space)
            form_1.quad_grid = 'gauss'
            K = MeshFunction(my_mesh)
            K.continous_tensor = [diff_tens_11, diff_tens_12, diff_tens_22]
            M_1 = inner(form, form_1, K)
            for el in range(n[i][0] * n[i][1]):
                npt.assert_array_almost_equal(
                    M_1_ref[:, el, :], M_1[:, :, el])

    # @unittest.skip
    def test_weighted_metric(self):
        # TODO: figure out why if the metric tensor is set to ones the result is very different
        """Compare weighted and unweighted metric terms with K set to identity."""
        mesh = CrazyMesh(2, (1, 1), ((-1, 1), (-1, 1)), curvature=0.2)
        K = MeshFunction(mesh)
        func_space = FunctionSpace(mesh, '1-lobatto', (3, 3))
        basis = BasisForm(func_space)
        K.continous_tensor = [diff_tens_11, diff_tens_12, diff_tens_22]
        xi = eta = np.linspace(-1, 1, 5)
        xi, eta = np.meshgrid(xi, eta)
        g_11_k, g_12_k, g_22_k = basis.weighted_metric_tensor(xi.ravel('F'), eta.ravel('F'), K)
        g_11, g_12, g_22 = mesh.metric_tensor(xi.ravel('F'), eta.ravel('F'))
        npt.assert_array_almost_equal(g_11, g_11_k)

    # @unittest.skip
    def test_weighted_inner_continous(self):
        """Test for weighted inner product."""
        mesh = CrazyMesh(2, (2, 2), ((-1, 1), (-1, 1)), curvature=0.2)
        func_space = FunctionSpace(mesh, '1-lobatto', (3, 4))
        basis = BasisForm(func_space)
        basis.quad_grid = 'gauss'
        K = MeshFunction(mesh)
        K.continous_tensor = [diff_tens_11, diff_tens_12, diff_tens_22]
        M_1_weighted = inner(basis, basis, K)
        M_1 = inner(basis, basis)
        npt.assert_array_almost_equal(M_1, M_1_weighted)


if __name__ == '__main__':
    unittest.main()
