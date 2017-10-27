"""Test for basis 2 forms."""
import path_magic
import unittest
import os
import sys
from mesh import CrazyMesh
from basis_forms import BasisForm
from function_space import FunctionSpace
import numpy as np
import numpy.testing as npt
from inner_product import inner


class TestBasis2Forms(unittest.TestCase):

    def setUp(self):
        self.mesh = CrazyMesh(
            2,  (2, 2), ((-1, 1), (-1, 1)), curvature=0.1)

    # @unittest.skip
    def test_value_face_basis(self):
        for p in range(2, 15):
            function_space = FunctionSpace(self.mesh, '2-lobatto', p)
            basis = BasisForm(function_space)
            basis.quad_grid = 'lobatto'
            ref_funcs = np.loadtxt(
                os.getcwd() + '/src/tests/test_basis_2_form/basis_2_form_p_' + str(p) + '.dat', delimiter=',')
            # decimals limited due to the limited precision of the matlab data
            npt.assert_array_almost_equal(
                ref_funcs, basis.basis, decimal=1)
    #

    # @unittest.skip
    def test_inner(self):
        list_cases = ['p2_n2-2', 'p2_n3-2',
                      'p5_n1-10', 'p10_n2-2', 'p13_n12-8']
        p = [2, 2, 5, 10, 13]
        n = [(2, 2), (3, 2), (1, 10), (2, 2), (12, 8)]
        curvature = [0.1, 0.1, 0.1, 0.1, 0.1]
        for i, case in enumerate(list_cases[:-1]):
            print("Test case n : ", i)
            # basis = basis_forms.
            # print("theo size ", (p[i] + 1)**2 *
            #       (p[i] + 1)**2 * n[i][0] * n[i][1])
            M_2_ref = np.loadtxt(
                os.getcwd() + '/src/tests/test_M_2/M_2_' + case + '.dat', delimiter=',').reshape((p[i])**2,  n[i][0] * n[i][1], (p[i])**2)
            # print("actual size ", np.size(M_0_ref))

            # print(np.shape(M_0_ref))
            my_mesh = CrazyMesh(
                2,  n[i], ((-1, 1), (-1, 1)), curvature=curvature[i])
            function_space = FunctionSpace(my_mesh, '2-lobatto', p[i])
            basis_0 = BasisForm(function_space)
            basis_0.quad_grid = 'lobatto'
            basis_1 = BasisForm(function_space)
            basis_1.quad_grid = 'lobatto'
            M_2 = inner(basis_0, basis_1)
            # print("REF ------------------ \n", M_2_ref[:, 0, :])
            # print("CALCULATED _----------------\n", M_2[:, :, 0])
            for el in range(n[i][0] * n[i][1]):
                npt.assert_array_almost_equal(
                    M_2_ref[:, el, :], M_2[:, :, el], decimal=7)


if __name__ == '__main__':
    unittest.main()
