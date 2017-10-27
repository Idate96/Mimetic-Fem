import path_magic
import unittest
from function_space import FunctionSpace, NextSpace
from mesh import CrazyMesh
from coboundaries import d, d_21_lobatto_outer, d_10_lobatto_inner, d_10_lobatto_outer
import numpy as np
from inner_product import inner
from basis_forms import BasisForm
import numpy.testing as npt
from forms import Form, cochain_to_global, cochain_to_local, ExtGaussForm_0
import matplotlib.pyplot as plt


class TestCoboundary(unittest.TestCase):

    def setUp(self):
        self.mesh = CrazyMesh(2, (2, 2), ((-1, 1), (-1, 1)), curvature=0)

    def pfun(self, x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def ufun(self):

        def ufun_x(x, y):
            return np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)

        def ufun_y(x, y):
            return np.sin(np.pi * x) * np.pi * np.cos(np.pi * y)

        return ufun_x, ufun_y

    def test_func_space_input(self):
        """Test function space input to coboudary operator.

        It should return the incidence matrix.
        """
        p_x, p_y = 3, 3
        func_space = FunctionSpace(self.mesh, '1-lobatto', (p_x, p_y), is_inner=False)
        e_21_lobatto_ref = d_21_lobatto_outer((p_x, p_y))
        e_21_lobatto = d(func_space)
        npt.assert_array_almost_equal(e_21_lobatto, e_21_lobatto_ref)

    def test_form_input(self):
        p_x, p_y = 3, 3
        func_space = FunctionSpace(self.mesh, '1-lobatto', (p_x, p_y), is_inner=False)
        form_1_cochain = np.random.rand((func_space.num_dof))

        form_2_cochain_local_ref = d_21_lobatto_outer(
            (p_x, p_y)) @ cochain_to_local(func_space, form_1_cochain)
        func_space_2 = NextSpace(func_space)
        form_2_cochain_ref = cochain_to_global(func_space_2, form_2_cochain_local_ref)

        form_1 = Form(func_space, form_1_cochain)

        form_2 = d(form_1)
        npt.assert_array_almost_equal(form_2_cochain_ref, form_2.cochain)

    # @unittest.skip
    def test_basis_input(self):
        """Test the coboundary operator with some basis as inputs."""
        p_x, p_y = 2, 2
        func_space_0 = FunctionSpace(self.mesh, '0-lobatto', (p_x, p_y), is_inner=False)
        func_space_1 = FunctionSpace(self.mesh, '1-lobatto', (p_x, p_y), is_inner=False)
        func_space_2 = FunctionSpace(self.mesh, '2-lobatto', (p_x, p_y), is_inner=False)
        basis_0_ref = BasisForm(func_space_0)
        basis_1_ref = BasisForm(func_space_1)
        basis_0_ref.quad_grid = 'lobatto'
        basis_1_ref.quad_grid = 'gauss'
        basis_2_ref = BasisForm(func_space_2)
        basis_2_ref.quad_grid = 'lobatto'
        e_21_ref = d_21_lobatto_outer((p_x, p_y))
        e_10_ref = d_10_lobatto_outer((p_x, p_y))
        basis_1, e_10 = d(basis_0_ref)
        basis_1.quad_grid = 'gauss'
        basis_2, e_21 = d(basis_1_ref)
        basis_2.quad_grid = 'lobatto'
        M_1 = inner(basis_1, basis_1)
        M_1_ref = inner(basis_1_ref, basis_1_ref)
        npt.assert_array_almost_equal(M_1_ref, M_1)
        M_2 = inner(basis_2, basis_2)
        M_2_ref = inner(basis_2_ref, basis_2_ref)
        npt.assert_array_almost_equal(M_2_ref, M_2)
        npt.assert_array_equal(e_21_ref, e_21)
        npt.assert_array_equal(e_10_ref, e_10)

    def test_coboundary_lobatto_e10_inner(self):
        """Test of the coundary 0->1 for lobatto form inner oriented."""

        p = (20, 20)
        n = (6, 6)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.1)
        xi = eta = np.linspace(-1, 1, 100)

        func_space = FunctionSpace(mesh, '0-lobatto', p)
        form_0 = Form(func_space)
        form_0.discretize(self.pfun)
        form_1 = d(form_0)
        form_1.reconstruct(xi, eta)
        (x, y), data_dx, data_dy = form_1.export_to_plot()

        ufun_x, ufun_y = self.ufun()
        npt.assert_array_almost_equal(ufun_x(x, y), data_dx)
        npt.assert_array_almost_equal(ufun_y(x, y), data_dy)

    # @unittest.skip
    def test_coboundary_extended_gauss_e10_inner(self):
        def pfun(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        p = (10, 10)
        n = (20, 20)

        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.3)
        func_space_extGau = FunctionSpace(mesh, '0-ext_gauss', p)

        form_0_extG = Form(func_space_extGau)
        form_0_extG.discretize(self.pfun)

        xi = eta = np.linspace(-1, 1, 10)
        form_0_extG.reconstruct(xi, eta)

        form_1 = d(form_0_extG)
        form_1.reconstruct(xi, eta)
        (x, y), data_dx, data_dy = form_1.export_to_plot()
        ufun_x, ufun_y = self.ufun()
        npt.assert_array_almost_equal(ufun_x(x, y), data_dx)
        npt.assert_array_almost_equal(ufun_y(x, y), data_dy)
        print(np.min(data_dy))

    @unittest.skip
    def test_visulalization_extended_gauss_e10_inner(self):
        def pfun(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        p = (10, 10)
        n = (20, 20)

        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.3)
        func_space_extGau = FunctionSpace(mesh, '0-ext_gauss', p)

        form_0_extG = Form(func_space_extGau)
        form_0_extG.discretize(self.pfun)

        xi = eta = np.linspace(-1, 1, 10)
        form_0_extG.reconstruct(xi, eta)
        (x, y), data = form_0_extG.export_to_plot()
        plt.contourf(x, y, data)
        plt.title('reduced extended_gauss 0-form')
        plt.colorbar()
        plt.show()

        form_1 = d(form_0_extG)
        form_1.reconstruct(xi, eta)
        (x, y), data_dx, data_dy = form_1.export_to_plot()
        ufun_x, ufun_y = self.ufun()
        plt.contourf(x, y, data_dx)
        plt.title('extended_gauss 1-form dx')
        plt.colorbar()
        plt.show()
        #
        plt.contourf(x, y, data_dy)
        plt.title('extended_gauss 1-form dy')
        plt.colorbar()
        plt.show()
        print(np.min(data_dy))

    @unittest.skip
    def test_visulization_lobato_e10_inner(self):

        def pfun(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)
        p = (20, 20)
        n = (2, 2)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.2)
        xi = eta = np.linspace(-1, 1, 100)

        func_space = FunctionSpace(mesh, '0-lobatto', p)
        form_0 = Form(func_space)
        form_0.discretize(pfun)
        form_0.reconstruct(xi, eta)
        (x, y), data = form_0.export_to_plot()
        plt.contourf(x, y, data)
        plt.title('reduced lobatto 0-form')
        plt.colorbar()
        plt.show()

        form_1 = d(form_0)

        form_1.reconstruct(xi, eta)
        (x, y), data_dx, data_dy = form_1.export_to_plot()

        plt.contourf(x, y, data_dx)
        plt.title('lobatto 1-form dx')
        plt.colorbar()
        # plt.show()

        plt.contourf(x, y, data_dy)
        plt.title('lobatto 1-form dy')
        plt.colorbar()
        # plt.show()
        print(np.min(data_dy))


if __name__ == '__main__':
    unittest.main()
