"""Test for basis forms."""
import path_magic
import os
import unittest
from mesh import CrazyMesh
from basis_forms import BasisForm
from function_space import FunctionSpace
import numpy as np
import numpy.testing as npt
from inner_product import inner


class TestMass_0_Matrix(unittest.TestCase):

    def test_inner(self):
        list_cases = ['p2_n2-2', 'p2_n3-2',
                      'p5_n1-10', 'p10_n2-2', 'p18_n14-14']
        p = [2, 2, 5, 10, 18]
        n = [(2, 2), (3, 2), (1, 10), (2, 2), (14, 10)]
        curvature = [0.2, 0.1, 0.2, 0.2, 0.2]
        for i, case in enumerate(list_cases[:-1]):
            print("Test case n : ", i)
            # basis = basis_forms.
            # print("theo size ", (p[i] + 1)**2 *
            #       (p[i] + 1)**2 * n[i][0] * n[i][1])
            M_0_ref = np.loadtxt(
                os.getcwd() + '/src/tests/test_M_0/M_0_' + case + '.dat', delimiter=',').reshape((p[i] + 1)**2,  n[i][0] * n[i][1], (p[i] + 1)**2)

            # print(np.shape(M_0_ref))
            my_mesh = CrazyMesh(
                2,  n[i], ((-1, 1), (-1, 1)), curvature=curvature[i])
            function_space = FunctionSpace(my_mesh, '0-lobatto', p[i])
            basis = BasisForm(function_space)
            basis.quad_grid = 'lobatto'
            basis_1 = BasisForm(function_space)
            basis_1.quad_grid = 'lobatto'
            M_0 = inner(basis, basis_1)
            for el in range(n[i][0] * n[i][1]):
                npt.assert_array_almost_equal(
                    M_0_ref[:, el, :], M_0[:, :, el], decimal=4)


if __name__ == '__main__':
    unittest.main()
