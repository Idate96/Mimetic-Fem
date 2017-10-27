"""this is a test case."""
import time
import path_magic
from mesh import TransfiniteMesh
from function_space import FunctionSpace
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
from quadrature import gauss_quad, extended_gauss_quad, lobatto_quad
from polynomials import lagrange_basis, edge_basis
from dof_map import dof_map_crazy_ext_gauss_nodes, dof_map_crazy_lobatto_edges_discontinous, dof_map_crazy_lobatto_faces

# np.set_printoptions(threshold=np.nan)


def d_21_lobatto_outer_virtual(p):
    """this program contains the incidence matrix with virtual elements implementing the neuman boundary conditions"""
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
    return virtual_E21


def assemble_virtual_E21(virtual_E21_3D):
    """this program assembles ghost E21"""
    virtual_E21_assembled = np.zeros((num_total_ghost_nodes, num_total_edges))
    for ele_num in range(num_total_elements):
        for test_dof in range(func_space_0_ext_gauss.num_internal_local_dof, func_space_0_ext_gauss.num_local_dof):
            for q_dof in range(num_local_element_edges):
                global_virtual_volume = dof_map_ext_gauss_nodes[ele_num,
                                                                test_dof] - num_total_elements * px**2
                global_q = dof_map_lobatto_edges_discontinous[ele_num, q_dof]
                virtual_E21_assembled[global_virtual_volume,
                                      global_q] = virtual_E21_3D[test_dof - px**2, q_dof, ele_num]
    return virtual_E21_assembled


def ghost_nodes_dof_map_boundary():
    dof_ghost_points_boundary_bottom = np.zeros((px * elements_layout[0]))
    dof_ghost_points_boundary_top = np.zeros((px * elements_layout[0]))
    dof_ghost_points_boundary_left = np.zeros((px * elements_layout[1]))
    dof_ghost_points_boundary_right = np.zeros((px * elements_layout[1]))

    for i in range(elements_layout[0]):
        # bottom boundary
        dof_ghost_points_boundary_bottom[i * px: (i + 1) * px] = \
            dof_map_ext_gauss_nodes[i * elements_layout[1], px**2 + 2 * px:px**2 + 3 * px]
        # top boundary
        dof_ghost_points_boundary_top[i * px: (i + 1) * px] = dof_map_ext_gauss_nodes[(
            i + 1) * elements_layout[1] - 1, px**2 + 3 * px:px**2 + 4 * px]

    for j in range(elements_layout[1]):
        # left boundary
        dof_ghost_points_boundary_left[j * px:(j + 1) * px] = \
            dof_map_ext_gauss_nodes[j, px**2: px**2 + px]
        # right boundary
        dof_ghost_points_boundary_right[j * px:(j + 1) * px] = dof_map_ext_gauss_nodes[(
            elements_layout[0] - 1) * elements_layout[1] + j, px**2 + px: px**2 + 2 * px]

    return dof_ghost_points_boundary_bottom, dof_ghost_points_boundary_top, \
        dof_ghost_points_boundary_left, dof_ghost_points_boundary_right


def wedge_2():
    lobatto_nodes, lobatto_weights = lobatto_quad(p[0])
    gauss_nodes, gauss_weights = gauss_quad(p[0] - 1)
    extended_gauss_nodes, extended_gauss_weights = extended_gauss_quad(p[0] - 1)

    hp3 = lagrange_basis(extended_gauss_nodes, gauss_nodes)
    ep1 = edge_basis(lobatto_nodes, gauss_nodes)

    W2_left = np.zeros((p[0], num_local_ghost_nodes))
    W2_right = np.zeros((p[0], num_local_ghost_nodes))
    W2_bottom = np.zeros((p[0], num_local_ghost_nodes))
    W2_top = np.zeros((p[0], num_local_ghost_nodes))

    """left - right edges"""
    for j in range(p[0]):
        nodeij = j
        for l in range(p[0]):
            edgekl_left = l
            edgekl_right = p[0] + l
            for q in range(p[0]):
                W2_left[nodeij, edgekl_left] = W2_left[nodeij, edgekl_left] + \
                    hp3[j + 1, q] * ep1[l, q] * gauss_weights[q]
                W2_right[nodeij, edgekl_right] = W2_right[nodeij, edgekl_right] + \
                    hp3[j + 1, q] * ep1[l, q] * gauss_weights[q]

    """bottom - top edges"""
    for i in range(p[0]):
        nodeij = i
        for k in range(p[0]):
            edgekl_bottom = 2 * p[0] + k
            edgekl_top = 3 * p[0] + k
            for q in range(p[0]):
                W2_bottom[nodeij, edgekl_bottom] = W2_bottom[nodeij, edgekl_bottom] + \
                    hp3[i + 1, q] * ep1[k, q] * gauss_weights[q]
                W2_top[nodeij, edgekl_top] = W2_top[nodeij, edgekl_top] + \
                    hp3[i + 1, q] * ep1[k, q] * gauss_weights[q]

    W2 = np.vstack((W2_left, W2_right, W2_bottom, W2_top))
    return W2


start_time = time.time()

dim = 2
elements_layout = (4, 4)

p = (4, 4)
p2 = (p[0] - 1, p[0] - 1)
px, py = p

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
    return -8 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def k_11():
    """Define the k11 component of permeability tensor."""
    return np.ones((1, elements_layout[0] * elements_layout[1]))


def k_12():
    """Define the k12 component of permeability tensor."""
    return np.zeros((1, elements_layout[0] * elements_layout[1]))


def k_22():
    """Define the k22 component of permeability tensor."""
    return np.ones((1, elements_layout[0] * elements_layout[1]))


def manufactured_solution(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


gamma = (gamma1, gamma2, gamma3, gamma4)
dgamma = (dgamma1, dgamma2, dgamma3, dgamma4)

ref_mesh = TransfiniteMesh(dim, elements_layout, gamma, dgamma)

func_space_0_ext_gauss = FunctionSpace(ref_mesh, '0-ext_gauss', p2, primal_is_inner)
func_space_1_lobatto = FunctionSpace(ref_mesh, '1-lobatto', p, primal_is_inner)
func_space_1_lobatto.dof_map.continous_dof = False
func_space_2_lobatto = FunctionSpace(ref_mesh, '2-lobatto', p, primal_is_inner)

anisotropic_tensor = MeshFunction(ref_mesh)
anisotropic_tensor.discrete_tensor = [k_11(), k_12(), k_22()]

source_form = Form(func_space_2_lobatto)
source_form.discretize(source)

phi_0_exact = Form(func_space_0_ext_gauss)
phi_0_exact.discretize(manufactured_solution)

basis_0 = BasisForm(func_space_0_ext_gauss)
basis_1 = BasisForm(func_space_1_lobatto)
basis_2 = BasisForm(func_space_2_lobatto)

basis_0.quad_grid = 'gauss'
basis_1.quad_grid = 'gauss'
basis_2.quad_grid = 'gauss'

# define forms for soultion using function space

q_1 = Form(func_space_1_lobatto)
q_1.basis.quad_grid = 'gauss'
phi_0 = Form(func_space_0_ext_gauss)
phi_0.basis.quad_grid = 'gauss'

"""general variables used frequently"""

num_total_elements = elements_layout[0] * elements_layout[1]
num_local_ghost_nodes = func_space_0_ext_gauss.num_local_dof - \
    func_space_0_ext_gauss.num_internal_local_dof
num_total_ghost_nodes = px * elements_layout[0] * (elements_layout[1] + 1) + py * elements_layout[1] * (
    elements_layout[0] + 1)
num_local_element_edges = 2 * px * (px + 1)
num_total_edges = func_space_1_lobatto.num_dof
num_total_surfaces = func_space_2_lobatto.num_dof
num_local_surfaces = func_space_2_lobatto.num_local_dof

dof_map_ext_gauss_nodes = func_space_0_ext_gauss.dof_map.dof_map
dof_map_lobatto_edges_discontinous = func_space_1_lobatto.dof_map.dof_map
dof_map_lobatto_faces = dof_map_ext_gauss_nodes

"""Define 1-form mass matrix"""
M_1 = inner(basis_1, basis_1)
M1_assembled = assemble(M_1, (func_space_1_lobatto, func_space_1_lobatto))

"""Define wedge 1"""
E21 = d_21_lobatto_outer(p)
W1 = basis_2.wedged(basis_0)

W1_E21 = np.dot(W1, E21)
W1_E21_3D = np.repeat(W1_E21[:, :, np.newaxis], num_total_elements, axis=2)

W1_E21_assembled = assemble(W1_E21_3D, (func_space_0_ext_gauss, func_space_1_lobatto))

"""Define Wedge 2"""
"""Define the second wedge / duality pairing / element connectivity matrix"""
virtual_E21 = d_21_lobatto_outer_virtual(p)
W2 = wedge_2()

W2_vE21 = np.dot(W2, virtual_E21)
W2_vE21_3D = np.repeat(W2_vE21[:, :, np.newaxis], num_total_elements, axis=2)

W2_vE21_assembled = assemble_virtual_E21(W2_vE21_3D)

"""assemble lhs"""
lhs = sparse.bmat([[M1_assembled, W1_E21_assembled.transpose(), W2_vE21_assembled.transpose()], [
    W1_E21_assembled, None, None], [W2_vE21_assembled, None, None]]).tolil()

"""assemble rhs"""
rhs_1 = np.zeros(num_total_edges)[:, np.newaxis]

local_f_cochain = np.zeros((num_local_surfaces, 1, num_total_elements))
local_W_f_3D = np.zeros((func_space_0_ext_gauss.num_internal_local_dof, 1, num_total_elements))

for ele_num in range(num_total_elements):
    for local_f in range(func_space_2_lobatto.num_local_dof):
        global_f = dof_map_lobatto_faces[ele_num, local_f]
        local_f_cochain[local_f, 0, ele_num] = source_form.cochain[global_f]
        # local_f_cochain = source_form.cochain[ele_num * func_space_2_lobatto.num_local_dof: (
        #     ele_num + 1) * func_space_2_lobatto.num_local_dof]
        # local_W_tmp = np.dot(W1, local_f_cochain)
        # local_W_f_3D[:, :, ele_num] = local_W_tmp[:, np.newaxis]

for ele_num in range(num_total_elements):
    local_W_f_3D[:, :, ele_num] = np.dot(W1, local_f_cochain[:, :, ele_num])

W_f_assembled = np.zeros((num_total_surfaces, 1))
for ele_num in range(num_total_elements):
    for local_f in range(func_space_2_lobatto.num_local_dof):
        global_surface = dof_map_lobatto_faces[ele_num, local_f]
        W_f_assembled[global_surface, 0] = local_W_f_3D[local_f, 0, ele_num]

rhs_2 = W_f_assembled

rhs_3 = np.zeros((num_total_ghost_nodes))[:, np.newaxis]

rhs = np.vstack((rhs_1, rhs_2, rhs_3))

"""implement boundary conditions"""
# importing ghost boundary points
dof_ghost_points_boundary_bottom, dof_ghost_points_boundary_top, \
    dof_ghost_points_boundary_left, dof_ghost_points_boundary_right = ghost_nodes_dof_map_boundary()
# print(dof_ghost_points_boundary_bottom, dof_ghost_points_boundary_top,
#       dof_ghost_points_boundary_left, dof_ghost_points_boundary_right)

lhs_dof_bottom = func_space_1_lobatto.num_dof + dof_ghost_points_boundary_bottom
lhs_dof_top = func_space_1_lobatto.num_dof + dof_ghost_points_boundary_top
lhs_dof_left = func_space_1_lobatto.num_dof + dof_ghost_points_boundary_left
lhs_dof_right = func_space_1_lobatto.num_dof + dof_ghost_points_boundary_right

lhs[lhs_dof_bottom] = 0
lhs[lhs_dof_left] = 0
lhs[lhs_dof_right] = 0
lhs[lhs_dof_top] = 0

lhs[lhs_dof_bottom, lhs_dof_bottom] = 1
lhs[lhs_dof_top, lhs_dof_top] = 1
lhs[lhs_dof_left, lhs_dof_left] = 1
lhs[lhs_dof_right, lhs_dof_right] = 1

# print(lhs_dof_bottom)
# print(lhs[lhs_dof_bottom, :])
print(np.linalg.det(lhs.todense()))

"""changing rhs"""

"""solving system of equations"""

solution = sparse.linalg.spsolve(lhs.tocsc(), rhs)

end_time = time.time()

print("The total time taken by the program is : ", end_time - start_time)

q_1.cochain = solution[:func_space_1_lobatto.num_dof]
# # phi_0_internal = solution[func_space_1_lobatto.num_dof:func_space_1_lobatto.num_dof +
# #                           func_space_0_ext_gauss.num_internal_local_dof]
#
# # plot sparsity pattern
# plt.figure(1)
# plt.spy(lhs)
# plt.show(block=False)
#
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

q_x_exact = 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
print(np.max(q_x_exact), np.min(q_x_exact))

plt.figure(3)
plt.contourf(x, y, q_x_exact)

plt.colorbar()
plt.show(block=False)

# phi_0.reconstruct(xi, eta)
# (x1, y1), phi_plot = phi_0.export_to_plot()

# plt.figure(4)
# plt.contourf(x1, y1, phi_plot)

plt.show()
