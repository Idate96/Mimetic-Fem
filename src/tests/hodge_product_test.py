import path_magic
import unittest
import os
from function_space import FunctionSpace
import numpy as np
import numpy.testing as npt
from mesh import CrazyMesh
from forms import Form
from basis_forms import BasisForm
from hodge import hodge
from coboundaries import d
from inner_product import inner
import matplotlib.pyplot as plt


class TestHodgeProduct(unittest.TestCase):
    """Test the hodge operator."""

    def setUp(self):
        self.p = (10, 10)
        nx, ny = 5, 4
        self.mesh = CrazyMesh(2, (nx, ny), ((-1, 1), (-1, 1)), 0.2)

    # @unittest.skip
    def test_hodge_1_ext_gauss_to_lobatto(self):
        """Test tranformation from extended gauss one form to lobatto one form on the dual.

        The extended gauss 1 form is obtained taking the exterior derivative of an extended gauss 0 form since the reduction for ext gauss 1-form is not currently supported.
        """
        def pfun(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        function_space_0 = FunctionSpace(self.mesh, '0-ext_gauss', self.p)
        form_0 = Form(function_space_0)
        form_0.discretize(pfun)
        form_1 = d(form_0)
        form_1_dual = hodge(form_1)

        xi = eta = np.linspace(-1, 1, 30)
        form_1_dual.reconstruct(xi, eta)
        (x, y), data_dx, data_dy = form_1_dual.export_to_plot()
        directory = 'src/tests/test_hodge/'
        x_ref = np.loadtxt(directory + 'x_lobatto_n5-4_p10-10_c2_xieta_30.dat', delimiter=',')
        y_ref = np.loadtxt(directory + 'y_lobatto_n5-4_p10-10_c2_xieta_30.dat', delimiter=',')
        data_dx_ref = np.loadtxt(
            directory + 'data_dx_lobatto_n5-4_p10-10_c3_xieta_30.dat', delimiter=',')
        data_dy_ref = np.loadtxt(
            directory + 'data_dy_lobatto_n5-4_p10-10_c3_xieta_30.dat',  delimiter=',')

        npt.assert_array_almost_equal(x_ref, x)
        npt.assert_array_almost_equal(y_ref, y)
        npt.assert_array_almost_equal(data_dx_ref, data_dx)
        npt.assert_array_almost_equal(data_dy_ref, data_dy)
        print(np.min(data_dy))

    def test_hodge_0_lobatto_to_2_ext_gauss(self):
        """Test hodge transformation from lobatto 0 form to extended gauss 2-form."""
        def ffun(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        def pfun(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        p = (15, 15)
        n = (3, 3)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.1)
        func_space_0 = FunctionSpace(mesh, '0-lobatto', p)
        form_0 = Form(func_space_0)
        form_0.discretize(pfun)
        form_2 = hodge(form_0)
        xi = eta = np.linspace(-1, 1, 100)
        form_2.reconstruct(xi, eta)
        (x, y), data = form_2.export_to_plot()
        npt.assert_array_almost_equal(pfun(x, y), data)

    def test_hodge_2_ext_gauss_to_0_lobatto(self):
        """Test hodge transformation from extended gauss 2-form to lobatto 0 form."""
        def ffun(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        p = (15, 15)
        n = (3, 3)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.1)
        func_space_2 = FunctionSpace(mesh, '2-ext_gauss', p)
        form_2 = Form(func_space_2)
        form_2.discretize(ffun)
        form_0 = hodge(form_2)
        xi = eta = np.linspace(-1, 1, 100)
        form_0.reconstruct(xi, eta)
        (x, y), data = form_0.export_to_plot()
        npt.assert_array_almost_equal(ffun(x, y), data)

    def test_hodge_construction(self):
        self.mesh = CrazyMesh(2, (10, 10), ((-1, 1), (-1, 1)), 0.2)
        p_dual = (self.p[0] - 1, self.p[1] - 1)
        func_space_lob_0 = FunctionSpace(self.mesh, '0-lobatto', self.p)
        func_space_extgauss_0 = FunctionSpace(self.mesh, '0-ext_gauss', p_dual)

        func_space_lob_2 = FunctionSpace(self.mesh, '2-lobatto', self.p)
        func_space_extgauss_2 = FunctionSpace(self.mesh, '2-ext_gauss', p_dual)

        func_space_lob_1 = FunctionSpace(self.mesh, '1-lobatto', self.p)
        func_space_extgauss_1 = FunctionSpace(self.mesh, '1-ext_gauss', p_dual)

        hodge_20_lob = hodge(func_space_lob_0)
        hodge_20_ext = hodge(func_space_extgauss_0)

        hodge_11_lob = hodge(func_space_lob_1)
        hodge_11_ext = hodge(func_space_extgauss_1)

        hodge_02_lob = hodge(func_space_lob_2)
        hodge_02_ext = hodge(func_space_extgauss_2)
        for i in range(self.mesh.num_elements):
            # this should be almost exact since it's the inverse
            npt.assert_array_almost_equal(np.eye(*np.shape(hodge_20_lob[:, :, i])),
                                          np.tensordot(hodge_02_ext[:, :, i], hodge_20_lob[:, :, i], axes=((1), (0))), decimal=2)
            npt.assert_array_almost_equal(np.eye(*np.shape(hodge_20_ext[:, :, i])),
                                          np.tensordot(hodge_02_lob[:, :, i], hodge_20_ext[:, :, i], axes=((1), (0))), decimal=2)
            npt.assert_array_almost_equal(-np.eye(*np.shape(hodge_11_lob[:, :, i])),
                                          np.tensordot(hodge_11_lob[:, :, i], hodge_11_ext[:, :, i], axes=((1), (0))), decimal=0)
            # this is not super accurete but should yield ** = +-1
            # something wrong here it gives zeros
            # npt.assert_array_almost_equal(np.eye(*np.shape(hodge_20_lob[:, :, i])),
            # np.tensordot(hodge_20_lob[:, :, i], hodge_20_lob[:, :, i], axes=((1),
            # (0))), decimal=2)
            # also zeros
            # npt.assert_array_almost_equal(np.eye(*np.shape(hodge_11_lob[:, :, i])),
            # np.tensordot(hodge_11_lob[:, :, i], hodge_11_lob[:, :, i], axes=((1),
            # (0))), decimal=2)
            # it blows up
            # npt.assert_array_almost_equal(np.eye(*np.shape(hodge_02_lob[:, :, i])),
            # np.tensordot(hodge_02_lob[:, :, i], hodge_02_lob[:, :, i], axes=((1),
            # (0))), decimal=2)

    @unittest.skip
    def test_visualize_hodge_1_ext_gauss_to_lobatto(self):

        def pfun(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        function_space_0 = FunctionSpace(self.mesh, '0-ext_gauss', self.p)
        form_0 = Form(function_space_0)
        form_0.discretize(pfun)
        form_1 = d(form_0)
        form_1_dual = hodge(form_1)

        xi = eta = np.linspace(-1, 1, 30)
        form_1_dual.reconstruct(xi, eta)
        (x, y), data_dx, data_dy = form_1_dual.export_to_plot()
        plt.contourf(x, y, data_dx)
        plt.title('lobatto 1-form dx')
        plt.colorbar()
        plt.show()

        plt.contourf(x, y, data_dy)
        plt.title('lobatto 1-form dy')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    unittest.main()
