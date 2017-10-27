"""Module to test mapping of compuational domain into an arbitrary one."""
import path_magic
import unittest
import os
import mesh
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as npt
# os.chdir(os.getcwd() + '/tests')


class TestMappingMesh22(unittest.TestCase):

    def setUp(self):
        n = 10
        self.nx, self.ny = 2, 2
        self.crazy_mesh = mesh.CrazyMesh(
            2, (self.nx, self.ny), ((-1, 1), (-1, 1)), curvature=0.1)
        self.xi = self.eta = np.linspace(-1, 1, n)
        self.xi, self.eta = np.meshgrid(self.xi, self.eta)
        self.dir = os.getcwd() + '/src/tests/'

    def test_deformed_grid_mapping(self):
        for i in range(self.nx * self.ny):
            x_ref = np.loadtxt(self.dir + 'test_mapping/x_reference_domain_cc1_el' + str(i) + '.dat',
                               delimiter=',')
            y_ref = np.loadtxt(self.dir + 'test_mapping/y_reference_domain_cc1_el' + str(i) + '.dat',
                               delimiter=',')
            x, y = self.crazy_mesh.mapping(self.xi, self.eta, i)
            npt.assert_array_almost_equal(x, x_ref, decimal=4)
            npt.assert_array_almost_equal(y, y_ref, decimal=4)

    def test_dx_dxi(self):
        for i in range(self.nx * self.ny):
            dx_dxi_ref = np.loadtxt(self.dir + 'test_dx_dxi/dxdxi_ref_domain_cc1_el' +
                                    str(i) + '.dat', delimiter=',')
            dx_dxi_crazy = self.crazy_mesh.dx_dxi(self.xi, self.eta, i)
            # print('element : ', i)
            # print(dx_dxi_crazy, dx_dxi_ref)
            npt.assert_array_almost_equal(dx_dxi_ref, dx_dxi_crazy, decimal=4)

    def test_dx_deta(self):
        for i in range(self.nx * self.ny):
            dx_dxi_ref = np.loadtxt(self.dir + 'test_dx_deta/dxdeta_ref_domain_cc1_el' +
                                    str(i) + '.dat', delimiter=',')
            dx_dxi_crazy = self.crazy_mesh.dx_deta(self.xi, self.eta, i)
            # print('element : ', i)
            # print(dx_dxi_crazy, dx_dxi_ref)
            npt.assert_array_almost_equal(dx_dxi_ref, dx_dxi_crazy, decimal=4)

    def test_dy_dxi(self):
        for i in range(self.nx * self.ny):
            dx_dxi_ref = np.loadtxt(self.dir + 'test_dy_dxi/dydxi_ref_domain_cc1_el' +
                                    str(i) + '.dat', delimiter=',')
            dx_dxi_crazy = self.crazy_mesh.dy_dxi(self.xi, self.eta, i)

            npt.assert_array_almost_equal(dx_dxi_ref, dx_dxi_crazy, decimal=4)

    def test_dy_deta(self):
        for i in range(self.nx * self.ny):
            dx_dxi_ref = np.loadtxt(self.dir + 'test_dy_deta/dydeta_ref_domain_cc1_el' +
                                    str(i) + '.dat', delimiter=',')
            dx_dxi_crazy = self.crazy_mesh.dy_deta(self.xi, self.eta, i)

            npt.assert_array_almost_equal(dx_dxi_ref, dx_dxi_crazy, decimal=4)


if __name__ == '__main__':
    unittest.main()
