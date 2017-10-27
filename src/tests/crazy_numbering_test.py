"""Module to test numbering scheme."""
import path_magic
import unittest
from function_space import FunctionSpace
import numpy as np
import numpy.testing as npt
from mesh import CrazyMesh


class NodalNumberingTest(unittest.TestCase):

    def setUp(self):
        pass

    # def test_22_elements(self):
    #     mesh = CrazyMesh(2, (2, 2), ((-1, 1), (-1, 1)))
    #     func_space = FunctionSpace(mesh, '0-lobatto', 2)
    #     ordering = func_space.dof_map.dof_map
    #     ref_ordering = np.array((0, 1, 2, 5, 6, 7, 10, 11, 12,
    #                              2, 3, 4, 7, 8, 9, 12, 13, 14,
    #                              10, 11, 12, 15, 16, 17, 20, 21, 22,
    #                              12, 13, 14, 17, 18, 19, 22, 23, 24)).reshape(4, 9)
    #
    #     npt.assert_array_equal(ordering, ref_ordering)
    #
    # def test_21_elements(self):
    #     mesh = CrazyMesh(2, (2, 1), ((-1, 1), (-1, 1)))
    #     func_space = FunctionSpace(mesh, '0-lobatto', 2)
    #     ordering = func_space.dof_map.dof_map
    #     ref_ordering = np.array((0, 1, 2, 3, 4, 5, 6, 7, 8,
    #                              6, 7, 8, 9, 10, 11, 12, 13, 14)).reshape(2, 9)
    #
    #     npt.assert_array_equal(ordering, ref_ordering)
    #
    # def test_12_elements(self):
    #     mesh = CrazyMesh(2, (1, 2), ((-1, 1), (-1, 1)))
    #     func_space = FunctionSpace(mesh, '0-lobatto', 2)
    #     ordering = func_space.dof_map.dof_map
    #     ref_ordering = np.array((0, 1, 2, 5, 6, 7, 10, 11, 12,
    #                              2, 3, 4, 7, 8, 9, 12, 13, 14)).reshape(2, 9)
    #
    #     npt.assert_array_equal(ordering, ref_ordering)

#
# class EdgeNumberingTest(unittest.TestCase):
#
#     def test_22_elements(self):
#         mesh = CrazyMesh(2, (2, 2), ((-1, 1), (-1, 1)))
#         func_space = FunctionSpace(mesh, '1-lobatto', 2)
#         ordering = func_space.dof_map.dof_map
#         ref_ordering = np.array((0, 1,   2,   5,   6,   7,  20,  21,  24,  25,  28,  29,
#                                  2,   3,   4,   7,  8,   9,  22,  23,  26,  27,  30,  31,
#                                  10,  11,  12,  15,  16,  17,  28,  29,  32,  33,  36,  37,
#                                  12,  13,  14,  17,  18,  19,  30,  31,  34,  35,  38,  39,)).reshape(4, 12)
#         npt.assert_array_equal(ordering, ref_ordering)
#
#     def test_21_elements(self):
#         mesh = CrazyMesh(2, (2, 1), ((-1, 1), (-1, 1)))
#         func_space = FunctionSpace(mesh, '1-lobatto', 2)
#         ordering = func_space.dof_map.dof_map
#         ref_ordering = np.array((0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17,
#                                  6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21)).reshape(2, 12)
#         npt.assert_array_equal(ordering, ref_ordering)
#
#     def test_12_elements(self):
#         mesh = CrazyMesh(2, (1, 2), ((-1, 1), (-1, 1)))
#         func_space = FunctionSpace(mesh, '1-lobatto', 2)
#         ordering = func_space.dof_map.dof_map
#         ref_ordering = np.array((0, 1, 2, 5, 6, 7, 10, 11, 14, 15, 18, 19,
#                                  2, 3, 4, 7, 8, 9, 12, 13, 16, 17, 20, 21)).reshape(2, 12)
#         npt.assert_array_equal(ordering, ref_ordering)


if __name__ == '__main__':
    unittest.main()
