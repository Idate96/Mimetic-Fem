import path_magic
import unittest
import os
from function_space import FunctionSpace
from mesh import CrazyMesh
import numpy as np
import numpy.testing as npt
from inner_product import inner
from forms import Form, ExtGaussForm_0
import matplotlib.pyplot as plt
from basis_forms import BasisForm


def func(x, y):
    return x + y


class TestForm0(unittest.TestCase):
    """Test case for the class of 0-forms."""

    def pfun(self, x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    # @unittest.skip
    def test_discretize_simple(self):
        """Simple discretization of 0 forms."""
        p_s = [(2, 2)]
        n = (2, 2)
        for p in p_s:
            mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.0)
            func_space = FunctionSpace(mesh, '0-lobatto', p)
            form_0 = Form(func_space)
            form_0.discretize(self.pfun)
            # values at x = +- 0.5 and y = +- 0.5
            ref_value = np.array((1, -1, -1, 1))
            npt.assert_array_almost_equal(ref_value, form_0.cochain_local[4])

    def test_gauss_projection(self):
        p = (10, 10)
        n = (5, 6)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.3)
        func_space = FunctionSpace(mesh, '0-gauss', p)
        form_0_gauss = Form(func_space)
        form_0_gauss.discretize(self.pfun)
        xi = eta = np.linspace(-1, 1, 50)
        form_0_gauss.reconstruct(xi, eta)
        (x, y), data = form_0_gauss.export_to_plot()
        npt.assert_array_almost_equal(self.pfun(x, y), data)

    def test_lobatto_projection(self):
        p = (10, 10)
        n = (5, 6)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.3)
        func_space = FunctionSpace(mesh, '0-lobatto', p)
        form_0 = Form(func_space)
        form_0.discretize(self.pfun)
        xi = eta = np.linspace(-1, 1, 20)
        form_0.reconstruct(xi, eta)
        (x, y), data = form_0.export_to_plot()
        npt.assert_array_almost_equal(self.pfun(x, y), data)

    def test_ext_gauss_projection(self):
        p = (10, 10)
        n = (5, 6)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.3)
        func_space_extGau = FunctionSpace(mesh, '0-ext_gauss', p)
        form_0_extG = ExtGaussForm_0(func_space_extGau)
        form_0_extG.discretize(self.pfun)
        xi = eta = np.linspace(-1, 1, 20)
        form_0_extG.reconstruct(xi, eta)
        (x, y), data = form_0_extG.export_to_plot()
        npt.assert_array_almost_equal(self.pfun(x, y), data)


if __name__ == '__main__':
    unittest.main()
