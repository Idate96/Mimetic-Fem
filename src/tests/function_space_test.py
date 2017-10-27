import path_magic
import unittest
import os
from function_space import FunctionSpace, DualSpace
import numpy as np
import numpy.testing as npt
from mesh import CrazyMesh
from forms import Form


class TestFunctionSpace(unittest.TestCase):

    def test_dual_space(self):
        """Test the generation of the dual space."""
        p = 3, 3
        mesh = CrazyMesh(2, (1, 1), ((-1, 1), (-1, 1)), curvature=0.2)
        func_space = FunctionSpace(mesh, '0-lobatto', p)
        dual_space_ref = FunctionSpace(mesh, '2-gauss', (p[0] - 1, p[1] - 1))
        dual_space = DualSpace(func_space, extend=False)
        self.assertEqual(dual_space_ref.form_type, dual_space.form_type,
                         msg="Form type is different")
        self.assertEqual(dual_space_ref.p, dual_space.p)

        func_space = FunctionSpace(mesh, '2-gauss', p)
        dual_space_ref = FunctionSpace(mesh, '0-lobatto', (p[0] + 1, p[1] + 1))
        dual_space = DualSpace(func_space, extend=False)
        self.assertEqual(dual_space_ref.form_type, dual_space.form_type,
                         msg="Form type is different")
        self.assertEqual(dual_space_ref.p, dual_space.p)

        func_space = FunctionSpace(mesh, '0-gauss', p)
        dual_space_ref = FunctionSpace(mesh, '2-lobatto', (p[0] + 1, p[1] + 1))
        dual_space = DualSpace(func_space, extend=False)
        self.assertEqual(dual_space_ref.form_type, dual_space.form_type,
                         msg="Form type is different")
        self.assertEqual(dual_space_ref.p, dual_space.p)

        func_space = FunctionSpace(mesh, '2-lobatto', p)
        dual_space_ref = FunctionSpace(mesh, '0-gauss', (p[0] - 1, p[1] - 1))
        dual_space = DualSpace(func_space, extend=False)
        self.assertEqual(dual_space_ref.form_type, dual_space.form_type,
                         msg="Form type is different")
        self.assertEqual(dual_space_ref.p, dual_space.p)

        func_space = FunctionSpace(mesh, '1-lobatto', p)
        dual_space_ref = FunctionSpace(mesh, '1-ext_gauss', (p[0] - 1, p[1] - 1))
        dual_space = DualSpace(func_space, extend=True)
        self.assertEqual(dual_space_ref.form_type, dual_space.form_type,
                         msg="Form type is different")
        self.assertEqual(dual_space_ref.p, dual_space.p)

        func_space = FunctionSpace(mesh, '1-ext_gauss', p)
        dual_space_ref = FunctionSpace(mesh, '1-lobatto', (p[0] + 1, p[1] + 1))
        dual_space = DualSpace(func_space, extend=True)
        self.assertEqual(dual_space_ref.form_type, dual_space.form_type,
                         msg="Form type is different")
        self.assertEqual(dual_space_ref.p, dual_space.p)

        func_space = FunctionSpace(mesh, '1-lobatto', p)
        dual_space_ref = FunctionSpace(mesh, '1-gauss', (p[0] - 1, p[1] - 1))
        dual_space = DualSpace(func_space, extend=False)
        self.assertEqual(dual_space_ref.form_type, dual_space.form_type,
                         msg="Form type is different")
        self.assertEqual(dual_space_ref.p, dual_space.p)

        func_space = FunctionSpace(mesh, '1-gauss', p)
        dual_space_ref = FunctionSpace(mesh, '1-lobatto', (p[0] + 1, p[1] + 1))
        dual_space = DualSpace(func_space, extend=False)
        self.assertEqual(dual_space_ref.form_type, dual_space.form_type,
                         msg="Form type is different")
        self.assertEqual(dual_space_ref.p, dual_space.p)


if __name__ == '__main__':
    unittest.main()
