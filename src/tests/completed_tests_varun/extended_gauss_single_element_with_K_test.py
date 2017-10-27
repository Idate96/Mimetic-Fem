"""this is a test case."""
import time
import path_magic
from mesh import TransfiniteMesh
from function_space import FunctionSpace, DualSpace
from inner_product import MeshFunction, inner
import numpy as np
from scipy.sparse import coo_matrix
from forms import Form
from basis_forms import BasisForm
from coboundaries import d, d_21_lobatto_outer
from assemble import assemble
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show

start_time = time.time()

dim = 2
elements_layout = (1, 1)

p = (20, 20)
p2 = (p[0] - 1, p[1] - 1)

primal_is_inner = False


def gamma1(s):
    """Parameterized BOTTOM boundary curve of the domain."""
    return s, 0 * np.ones(np.shape(s))


def gamma2(t):
    """Parameterized RIGHT boundary curve of the domain."""
    return 1 * np.ones(np.shape(t)), t


def gamma3(s):
    """Parameterized TOP boundary curve of the domain."""
    return s, 1 * np.ones(np.shape(s))


def gamma4(t):
    """Parameterized LEFT boundary curve of the domain."""
    return 0 * np.ones(np.shape(t)), t


def dgamma1(s):
    """Differential of gamma1 wrt to xi."""
    return 1 * np.ones(np.shape(s)), 0 * np.ones(np.shape(s))


def dgamma2(t):
    """Differential of gamma2 wrt to eta."""
    return 0 * np.ones(np.shape(t)), 1 * np.ones(np.shape(t))


def dgamma3(s):
    """Differential of gamma3 wrt to xi."""
    return 1 * np.ones(np.shape(s)), 0 * np.ones(np.shape(s))


def dgamma4(t):
    """Differential of gamma4 wrt to eta."""
    return 0 * np.ones(np.shape(t)), 1 * np.ones(np.shape(t))


def source(x, y):
    """Define the source term."""
    # return -8 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    return -36 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    # return -36 * np.pi ** 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) + 24 * np.pi ** 2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)


def k_11():
    """Define the k11 component of permeability tensor."""
    return 4 * np.ones((1, 1))
    # return 4 * np.ones((1, 1))
    # return 4 * np.ones((1, 1))


def k_12():
    """Define the k12 component of permeability tensor."""
    return 0 * np.ones((1, 1))
    # return 3 * np.ones((1, 1))
    # return 3 * np.ones((1, 1))


def k_22():
    """Define the k22 component of permeability tensor."""
    return 5 * np.ones((1, 1))
    # return 5 * np.ones((1, 1))
    # return 5 * np.ones((1, 1))


gamma = (gamma1, gamma2, gamma3, gamma4)
dgamma = (dgamma1, dgamma2, dgamma3, dgamma4)

ref_mesh = TransfiniteMesh(dim, elements_layout, gamma, dgamma)

func_space_1_lobatto = FunctionSpace(ref_mesh, '1-lobatto', p, primal_is_inner)
func_space_1_lobatto.dof_map.continous_dof = False

func_space_0_ext_gauss = FunctionSpace(ref_mesh, '0-ext_gauss', p2, primal_is_inner)

func_space_2_lobatto = FunctionSpace(ref_mesh, '2-lobatto', p, primal_is_inner)
func_space_0_ext_gauss = DualSpace(func_space_2_lobatto)
anisotropic_tensor = MeshFunction(ref_mesh)
anisotropic_tensor.discrete_tensor = [k_11(), k_12(), k_22()]

source_form = Form(func_space_2_lobatto)
source_form.discretize(source)

# define basis forms from function space

basis_0 = BasisForm(func_space_0_ext_gauss)
basis_1 = BasisForm(func_space_1_lobatto)
basis_2 = BasisForm(func_space_2_lobatto)

basis_0.quad_grid = 'gauss'
basis_1.quad_grid = 'lobatto'
basis_2.quad_grid = 'lobatto'

# define forms for soultion using function space

q_1 = Form(func_space_1_lobatto)
# q_1.basis.quad_grid = 'lobatto'
phi_0 = Form(func_space_0_ext_gauss)
# phi_0.basis.quad_grid = 'gauss'

E21 = d_21_lobatto_outer(p)

"""extended E21 with virtual volumes"""

px, py = p

virtual_E21 = np.zeros((px * 4, 2 * px * (px + 1)))

for j in range(px):
    vol_id = px**2 + j
    edge_id = px * (px + 1) + j
    virtual_E21[vol_id - px**2, edge_id] = 1
    virtual_E21[vol_id - px**2 + px, edge_id + px**2] = -1

for i in range(px):
    vol_id = px**2 + 2 * px + i
    edge_id = (px + 1) * i
    virtual_E21[vol_id - px**2, edge_id] = 1
    virtual_E21[vol_id - px**2 + px, edge_id + px] = -1

# define the mass and wedge matrices

# anisotropic_tensor =
M_1 = inner(basis_1, basis_1, anisotropic_tensor)
M1_assembled = assemble(M_1, (func_space_1_lobatto, func_space_1_lobatto))

W_i_2 = basis_2.wedged(basis_0)
Wi_2_E_21 = np.dot(W_i_2, E21)

lhs = sparse.bmat([[M1_assembled, Wi_2_E_21.transpose(), virtual_E21.transpose()], [
                  Wi_2_E_21, None, None], [virtual_E21, None, None]]).tolil()

rhs_1 = np.zeros(2 * px * (px + 1))
rhs_2 = np.dot(W_i_2, source_form.cochain)
rhs_3 = np.zeros(4 * px)
rhs = np.hstack((rhs_1, rhs_2, rhs_3))

# implementing BC's

nr_ghost_points = px * 4
dof_ghost_points_boundary = range(2 * px * (px + 1) + px**2, 2 * px * (px + 1) + px**2 + px * 4)

# implementing dirichlet boundary conditions

# changing lhs
lhs[-nr_ghost_points:, :] = np.zeros((nr_ghost_points, np.shape(lhs)[1]))
lhs[dof_ghost_points_boundary, dof_ghost_points_boundary] = 1
print(np.linalg.det(lhs.todense()))

# changing rhs - do something here... !

solution = sparse.linalg.spsolve(lhs.tocsc(), rhs)

end_time = time.time()

print("The total time taken by the program is : ", end_time - start_time)

q_1.cochain = solution[:func_space_1_lobatto.num_dof]
# phi_0_internal = solution[func_space_1_lobatto.num_dof:func_space_1_lobatto.num_dof +
#                           func_space_0_ext_gauss.num_internal_local_dof]

# plot sparsity pattern
plt.figure(1)
plt.spy(lhs)
plt.show(block=False)

xi = eta = np.linspace(-1, 1, 30)

q_1.reconstruct(xi, eta)
(x, y), u_x, u_y = q_1.export_to_plot()
q_x = u_y
q_y = -u_x
print(np.max(q_x), np.min(q_x))

plt.figure(2)
plt.contourf(x, y, q_x)
plt.colorbar()
plt.show(block=False)

# q_x_exact = 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
# q_x_exact = 8 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) + \
#     6 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
q_x_exact = 8 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
# q_x_exact = 10 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
print(np.max(q_x_exact), np.min(q_x_exact))

plt.figure(3)
plt.contourf(x, y, q_x_exact)

plt.colorbar()
plt.show(block=False)

# plt.show()
