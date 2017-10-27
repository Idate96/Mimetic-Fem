import path_magic
import unittest
import os
from mesh import CrazyMesh
from function_space import FunctionSpace
import numpy as np
import numpy.testing as npt
from mesh import CrazyMesh
from forms import Form
from basis_forms import BasisForm
from coboundaries import d
from inner_product import inner


class TestInnerProduct(unittest.TestCase):
    """Test the inner product function."""

    def setUp(self):
        self.mesh = CrazyMesh(2, (2, 2), ((-1, 1), (-1, 1)), curvature=0)

    def test_basis_inner(self):
        """Test inner product with basis functions."""
        p_x, p_y = 2, 2
        func_space_0 = FunctionSpace(self.mesh, '0-lobatto', (p_x, p_y))
        func_space_1 = FunctionSpace(self.mesh, '1-lobatto', (p_x, p_y))
        basis_0 = BasisForm(func_space_0)
        basis_1 = BasisForm(func_space_1)
        basis_1.quad_grid = 'lobatto'
        basis_0.quad_grid = 'lobatto'
        M_1 = inner(basis_1, basis_1)
        e_10 = d(func_space_0)
        inner_prod_ref = np.tensordot(e_10, M_1, axes=((0), (0)))

        #
        inner_prod = inner(d(basis_0), basis_1)
        npt.assert_array_almost_equal(inner_prod_ref, inner_prod)
        #
        inner_prod_1_ref = np.rollaxis(np.tensordot(M_1, e_10, axes=((0), (0))), 2)

        inner_prod_1 = inner(basis_1, d(basis_0))
        npt.assert_array_almost_equal(inner_prod_1_ref, inner_prod_1)

        #
        inner_prod_2_ref = np.tensordot(e_10, np.tensordot(
            e_10, M_1, axes=((0), (0))), axes=((0), (1)))
        inner_prod_2 = inner(d(basis_0), d(basis_0))

        #
        npt.assert_array_almost_equal(inner_prod_2_ref, inner_prod_2)


if __name__ == '__main__':
    unittest.main()
