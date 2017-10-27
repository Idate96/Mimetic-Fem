import path_magic
import unittest
import os
from mesh import CrazyMesh, TransfiniteMesh
from function_space import FunctionSpace
import numpy as np
import numpy.testing as npt
from mesh import CrazyMesh
from forms import Form
from basis_forms import BasisForm
from coboundaries import d
from inner_product import inner


class TransfiniteTest(unittest.TestCase):

    def test_metric(self):

        def gamma1(s): return 2 * s - 1, -1 * np.ones(np.shape(s))

        def gamma2(t): return 1 * np.ones(np.shape(t)), 2 * t - 1

        def gamma3(s): return 2 * s - 1, 1 * np.ones(np.shape(s))

        def gamma4(t): return -1 * np.ones(np.shape(t)), 2 * t - 1

        def dgamma1(s): return 2 * np.ones(np.shape(s)), 0 * np.ones(np.shape(s))

        def dgamma2(t): return 0 * np.ones(np.shape(t)), 2 * np.ones(np.shape(t))

        def dgamma3(s): return 2 * np.ones(np.shape(s)), 0 * np.ones(np.shape(s))

        def dgamma4(t): return 0 * np.ones(np.shape(t)), 2 * np.ones(np.shape(t))

        gamma = (gamma1, gamma2, gamma3, gamma4)
        dgamma = (dgamma1, dgamma2, dgamma3, dgamma4)

        sand_shale_mesh = TransfiniteMesh(2, (1, 1), gamma, dgamma)
        crazy_mesh = CrazyMesh(2, (1, 1), ((-1, 1), (-1, 1)))
        xi = eta = np.linspace(-1, 1, 5)
        npt.assert_array_almost_equal(crazy_mesh.dx_dxi(xi, eta), sand_shale_mesh.dx_dxi(xi, eta))
        npt.assert_array_almost_equal(crazy_mesh.dx_deta(xi, eta), sand_shale_mesh.dx_deta(xi, eta))
        npt.assert_array_almost_equal(crazy_mesh.dy_dxi(xi, eta), sand_shale_mesh.dy_dxi(xi, eta))
        npt.assert_array_almost_equal(crazy_mesh.dy_deta(xi, eta), sand_shale_mesh.dy_deta(xi, eta))
        npt.assert_array_almost_equal(crazy_mesh.g(xi, eta), sand_shale_mesh.g(xi, eta))


if __name__ == '__main__':
    unittest.main()
