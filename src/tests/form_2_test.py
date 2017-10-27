import path_magic
import unittest
import os
from function_space import FunctionSpace
import numpy as np
import numpy.testing as npt
from mesh import CrazyMesh
from forms import Form
import matplotlib.pyplot as plt


def paraboloid(x, y):
    """Funcion used for testing."""
    return x**2 + y**2


class TestForm2(unittest.TestCase):
    """Test class of 2-forms."""
    @unittest.skip
    def test_discretize_lobatto(self):
        """Test the discretize method of the 2 - forms."""
        my_mesh = CrazyMesh(2, (2, 2), ((-1, 1), (-1, 1)), curvature=0)
        p_int_0 = 2
        for p in range(3, 8):
            filename = 'discrete_2_form_p' + \
                str(p) + '_p_int' + str(p_int_0) + '.dat'
            directory = os.getcwd() + '/src/tests/test_discrete_2_form/'
            func_space = FunctionSpace(my_mesh, '2-lobatto', p)
            form = Form(func_space)
            form.discretize(paraboloid, ('gauss', p_int_0))
            cochain_ref = np.loadtxt(directory + filename, delimiter=',')
            npt.assert_array_almost_equal(form.cochain_local, cochain_ref, decimal=4)
        p = 5
        for p_int in range(2, 6):
            filename = 'discrete_2_form_p' + \
                str(p) + '_p_int' + str(p_int_0) + '.dat'
            directory = os.getcwd() + '/src/tests/test_discrete_2_form/'
            func_space = FunctionSpace(my_mesh, '2-lobatto', p)
            form = Form(func_space)
            form.discretize(paraboloid, ('gauss', p_int_0))
            cochain_ref = np.loadtxt(directory + filename, delimiter=',')
            npt.assert_array_almost_equal(form.cochain_local, cochain_ref, decimal=4)

        p = 6
        p_int = 5
        filename_1 = 'discrete_2_form_p' + \
            str(p) + '_p_int' + str(p_int) + '_el_57.dat'
        directory = os.getcwd() + '/src/tests/test_discrete_2_form/'
        mesh_1 = CrazyMesh(2, (5, 7), ((-1, 1), (-1, 1)), curvature=0)
        func_space = FunctionSpace(mesh_1, '2-lobatto', p)
        form = Form(func_space)
        form.discretize(paraboloid, ('gauss', p_int))
        cochain_ref_1 = np.loadtxt(directory + filename_1, delimiter=',')
        npt.assert_array_almost_equal(form.cochain_local, cochain_ref_1)

        mesh_2 = CrazyMesh(2, (5, 7), ((-1, 1), (-1, 1)), curvature=0.2)
        filename_2 = 'discrete_2_form_p' + \
            str(p) + '_p_int' + str(p_int) + '_el_57_cc2.dat'
        directory = os.getcwd() + '/src/tests/test_discrete_2_form/'
        func_space = FunctionSpace(mesh_2, '2-lobatto', p)
        form = Form(func_space)
        form.discretize(paraboloid, ('gauss', p_int))
        cochain_ref_2 = np.loadtxt(directory + filename_2, delimiter=',')
        npt.assert_array_almost_equal(form.cochain_local, cochain_ref_2)

    @unittest.skip
    def test_local_to_global_cochain(self):
        # TODO: currently not supported px != py
        """Test local to global mapping of cochians."""
        mesh = CrazyMesh(2, (2, 2), ((-1, 1), (-1, 1)), curvature=0.3)
        cases = ['2-lobatto', '2-gauss']
        p = 2, 3
        for form_case in cases:
            func_space = FunctionSpace(mesh, form_case, p)
            form_2 = Form(func_space)
            form_2.discretize(paraboloid)
            # np.assert_array_almost_equal(self.cochain, form_2.cochain)

    @unittest.skip
    def test_reconstruct_lobatto(self):
        """Test the reconstructin of lobatto forms."""
        p = (20, 20)
        n = (3, 2)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.3)
        xi = eta = np.linspace(-1, 1, 30)
        func_space = FunctionSpace(mesh, '2-lobatto', p)
        form_2 = Form(func_space)
        cochain = np.loadtxt(
            'src/tests/test_form_2/cochain_lobatto_n3-2_p20-20_c3.dat', delimiter=',')
        ref_reconstructed = np.loadtxt(
            'src/tests/test_form_2/reconstructed_lobatto_n3-2_p20-20_c3_xieta_30.dat', delimiter=',')
        form_2.cochain = cochain
        form_2.reconstruct(xi, eta)
        npt.assert_array_almost_equal(ref_reconstructed, form_2.reconstructed)

    @unittest.skip
    def test_project_lobatto(self):
        directory = 'src/tests/test_form_2/'
        x_ref = np.loadtxt(directory + 'x_lobatto_n3-2_p20-20_c3_xieta_30.dat', delimiter=',')
        y_ref = np.loadtxt(directory + 'y_lobatto_n3-2_p20-20_c3_xieta_30.dat', delimiter=',')
        data_ref = np.loadtxt(directory + 'data_lobatto_n3-2_p20-20_c3_xieta_30.dat', delimiter=',')

        def ffun(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)
        p = (20, 20)
        n = (3, 2)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.3)
        xi = eta = np.linspace(-1, 1, 30)

        func_space = FunctionSpace(mesh, '2-lobatto', p)
        form_2 = Form(func_space)
        form_2.discretize(ffun)
        form_2.reconstruct(xi, eta)
        (x, y), data = form_2.export_to_plot()
        npt.assert_array_almost_equal(x_ref, x)
        npt.assert_array_almost_equal(y_ref, y)
        npt.assert_array_almost_equal(data_ref, data)
    @unittest.skip
    def test_projection_gauss(self):
        def ffun(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        p = (20, 20)
        n = (3, 3)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.3)
        func_space = FunctionSpace(mesh, '2-ext_gauss', p)
        f2eg = Form(func_space)
        f2eg.discretize(ffun)
        xi = eta = np.linspace(-1, 1, 50)
        f2eg.reconstruct(xi, eta)
        (x, y), data = f2eg.export_to_plot()
        npt.assert_array_almost_equal(ffun(x, y), data)

#    @unittest.skip
    def test_reduction_visualization_lobatto_2form(self):
        def ffun(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)
        p = (20, 20)
        n = (2, 2)
        mesh = CrazyMesh(2, n, ((-1.25, 1), (-1.25, 1)), 0.3)
        xi = eta = np.linspace(-1, 1, 30)

        func_space = FunctionSpace(mesh, '2-lobatto', p)
        form_2 = Form(func_space)
        form_2.discretize(ffun)
        # np.savetxt('cochain_lobatto_n3-2_p20-20_c3.dat', form_2.cochain, delimiter=',')
        form_2.reconstruct(xi, eta)
        (x, y), data = form_2.export_to_plot()
        plt.contourf(x, y, data)
        plt.title('our reduced lobatto 2-form')
        plt.colorbar()
        plt.show()

    @unittest.skip
    def test_reduction_visualization_extended_gauss_2form(self):
        def ffun(x, y):
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        p = (20, 20)
        n = (3, 3)
        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.3)
        func_space = FunctionSpace(mesh, '2-ext_gauss', p)
        f2eg = Form(func_space)
        f2eg.discretize(ffun)
        xi = eta = np.linspace(-1, 1, 50)
        f2eg.reconstruct(xi, eta)
        (x, y), data = f2eg.export_to_plot()
        plt.contourf(x, y, data)
        plt.title('reduced extened_gauss 2-form')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    unittest.main()
