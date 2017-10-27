import path_magic
import unittest
import os
from function_space import FunctionSpace
import numpy as np
import numpy.testing as npt
from mesh import CrazyMesh
from forms import Form
import matplotlib.pyplot as plt


class TestForm1(unittest.TestCase):

    def ufun_x(self, x, y):
        func_dx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        return func_dx

    def ufun_y(self, x, y):
        func_dy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
        return func_dy

    @unittest.skip
    def test_discretize_lobatto(self):
        def func_x(x, y):
            return np.ones(np.shape(x))

        def func_y(x, y):
            return np.ones(np.shape(x)) * 2

        mesh = CrazyMesh(2, (2, 2), ((-1, 1), (-1, 1)), 0.3)
        func_space = FunctionSpace(mesh, '1-lobatto', (3, 3))
        form_1 = Form(func_space)
        form_1.discretize((func_x, func_y))
        cochain_local_ref = np.loadtxt(
            'src/tests/test_discrete_1_form/discrete_1_form_dx_1_dy_2.csv', delimiter=',')
        npt.assert_array_almost_equal(form_1.cochain_local, cochain_local_ref, decimal=3)

    @unittest.skip
    def test_projection_lobatto(self):
        p = (10, 10)
        n = (5, 6)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.1)
        func_space = FunctionSpace(mesh, '1-lobatto', p)
        form_1 = Form(func_space)
        form_1.discretize((self.ufun_x, self.ufun_y))
        xi = eta = np.linspace(-1, 1, 40)
        form_1.reconstruct(xi, eta)
        (x, y), data_dx, data_dy = form_1.export_to_plot()
        npt.assert_array_almost_equal(self.ufun_x(x, y), data_dx)
        npt.assert_array_almost_equal(self.ufun_y(x, y), data_dy)

    @unittest.skip
    def test_projection_gauss(self):
        p = (10, 10)
        n = (5, 6)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.1)
        func_space = FunctionSpace(mesh, '1-gauss', p)
        form_1 = Form(func_space)
        form_1.discretize((self.ufun_x, self.ufun_y))
        xi = eta = np.linspace(-1, 1, 40)
        form_1.reconstruct(xi, eta)
        (x, y), data_dx, data_dy = form_1.export_to_plot()
        npt.assert_array_almost_equal(self.ufun_x(x, y), data_dx, decimal=4)
        npt.assert_array_almost_equal(self.ufun_y(x, y), data_dy, decimal=4)

    def test_l2_norm(self):
        p = (10, 10)
        n = (10, 10)
        mesh = CrazyMesh(2, n, ((0, 1), (0, 1)), 0.1)
        func_space = FunctionSpace(mesh, '1-lobatto', p)
        form_1 = Form(func_space)
        form_1.discretize((self.ufun_x, self.ufun_y))
        # print("chochain : \n", form_1.chochain)
        error_global, error_local = form_1.l_2_norm((self.ufun_x, self.ufun_y))
        self.assertLess(error_global, 4 * 10**(-12))
        self.assertLess(error_local, 4 * 10**(-10))

    @unittest.skip
    def test_reduction_visualization_lobatto_1form(self):
        """Simple discretization of 1 forms."""
        p_s = [(10, 10)]
        n = (2, 2)
        for p in p_s:
            mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0)
            func_space = FunctionSpace(mesh, '1-lobatto', p)
            form_1 = Form(func_space)
            form_1.discretize((self.ufun_x, self.ufun_y))
            print(form_1.function_space.dof_map.dof_map)
            xi = eta = np.linspace(-1, 1, 40)
            form_1.reconstruct(xi, eta)
            (x, y), data_dx, data_dy = form_1.export_to_plot()

            plt.contourf(x, y, data_dx)
            plt.title('projected lobatto 1-form dx')
            plt.colorbar()
            plt.show()

            plt.contourf(x, y, data_dy)
            plt.title('projected lobatto 1-form dy')
            plt.colorbar()
            plt.show()
            print(np.min(data_dy))


if __name__ == '__main__':
    unittest.main()
