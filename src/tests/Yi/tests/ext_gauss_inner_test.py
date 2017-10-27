# -*- coding: utf-8 -*-
"""
(SHORT NAME EXPLANATION)
"""
from mesh import CrazyMesh
from function_space import FunctionSpace
from forms import Form
from inner_product import inner


mesh = CrazyMesh( 2, (2, 2), ((-1, 1), (-1, 1)), 0.1 )
func_space_eg0 = FunctionSpace(mesh, '0-ext_gauss', (5, 5))
f0  = Form(func_space_eg0)
#f0.basis.quad_grid = ('gauss',6)
egM0 = inner(f0.basis, f0.basis)
