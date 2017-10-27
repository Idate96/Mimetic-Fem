import src.tests.path_magic
from mesh import CrazyMesh
from function_space import FunctionSpace
from coboundaries import d_21_lobatto_outer

if __name__ == '__main__':
    print("hello world")

# mesh arguments

dim = 2
elements_layout = (1, 1)
bounds_domain = ((-1, 1), (-1, 1))
curvature = 0.1

# function space arguments

form_type = '0-lobatto'
p = (3, 3)
is_inner = False
numbering_rule = 'general'


crazy_mesh = CrazyMesh(dim, elements_layout, bounds_domain, curvature)

function_space_2_lobatto = FunctionSpace(crazy_mesh, '2-lobatto', p, is_inner)
function_space_1_lobatto = FunctionSpace(crazy_mesh, '1-lobatto', p, is_inner)

E_21_lobatto_outer = d_21_lobatto_outer(p)
