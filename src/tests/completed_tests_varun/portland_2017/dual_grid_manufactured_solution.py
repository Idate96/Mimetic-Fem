"""dual grid method"""

import time
import path_magic
from mesh import TransfiniteMesh
from function_space import FunctionSpace
from inner_product import MeshFunction, inner
import numpy as np
import scipy.io
import numpy.testing as npt
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
from my_functions import assemble_cochain, assemble_cochain2, mass_matrix_1_form_local

start_time = time.time()

"""define source term"""


# def source(x, y): return -8 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
# def source(x, y): return -36 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
def source(x, y): return -36 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 *
                                                                         np.pi * y) + 24 * np.pi**2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

"""define anisotropic tensor"""


def k_11(): return np.ones((1, element_layout[0] * element_layout[1]))


def k_12(): return np.zeros((1, element_layout[0] * element_layout[1]))


def k_22(): return np.ones((1, element_layout[0] * element_layout[1]))

"""define manufactured solution"""


def manufactured_solution(x, y): return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


"""define gammas"""


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


"""define dgammas"""


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


"""define dimension and element layout"""

dim = 2
element_layout = (5, 5)

"""define polynomial degree and inner/outer orientation"""

pp = (5, 5)                     # polynomial degree - primal mesh
pd = (pp[0] - 1, pp[1] - 1)     # polynomial degree - dual mesh

orientation_inner = True
outer = False
# is_inner = False        # orientation of primal mesh

"""define mesh"""
gamma = (gamma1, gamma2, gamma3, gamma4)
dgamma = (dgamma1, dgamma2, dgamma3, dgamma4)

mesh1 = TransfiniteMesh(dim, element_layout, gamma, dgamma)

"""define function spaces used in problem"""

fs_2_lobatto = FunctionSpace(mesh1, '2-lobatto', pp, outer)
fs_1_lobatto = FunctionSpace(mesh1, '1-lobatto', pp, outer)
fs_0_gauss = FunctionSpace(mesh1, '0-gauss', pd, inner)

fs_1_lobatto.dof_map.continous_dof = True          # continuous elements

"""define forms"""

"""define (n) - source form"""
f_source = Form(fs_2_lobatto)           # form for source term
f_source.discretize(source)
f_source.basis.quad_grid = 'gauss'

"""define (n-1) - q form"""
f_flux = Form(fs_1_lobatto)             # form for flux terms
f_flux.basis.quad_grid = 'lobatto'

"""define exact 0 - \phi form"""
f_phi_exact = Form(fs_0_gauss)
f_phi_exact.discretize(manufactured_solution)
f_phi_exact.basis.quad_grid = 'gauss'

"""define unkown 0 - \phi form"""
f_phi = Form(fs_0_gauss)
f_phi.basis.quad_grid = 'gauss'

"""define  anisotropic tensor as a mesh property"""
anisotropic_tensor = MeshFunction(mesh1)
anisotropic_tensor.discrete_tensor = [k_11(), k_12(), k_22()]

"""define basis functions"""
basis_2 = BasisForm(fs_2_lobatto)
basis_1 = BasisForm(fs_1_lobatto)
basis_0 = BasisForm(fs_0_gauss)

basis_2.quad_grid = 'gauss'
basis_1.quad_grid = 'lobatto'
basis_0.quad_grid = 'gauss'

"""general variables used frequently"""
num_total_elements = element_layout[0] * element_layout[1]
num_total_edges = fs_1_lobatto.num_dof
num_total_faces = fs_2_lobatto.num_dof
num_local_surfaces = fs_2_lobatto.num_local_dof
dof_map_lobatto_faces = fs_2_lobatto.dof_map.dof_map

"""define 1-form mass matrix"""

# M1 = inner(basis_1, basis_1)
# print(np.shape(M1))


# print(np.shape(M1_assembled))

# print(M1_assembled.todense())

lobatto_nodes, wp = lobatto_quad(pp[0])
gauss_nodes, wp = gauss_quad(pp[0] - 1)

# print(np.shape(lobatto_nodes))
# print(np.shape(gauss_nodes))
xi_1D, eta_1D = np.meshgrid(lobatto_nodes, lobatto_nodes)

xi = xi_1D.ravel('F')
eta = eta_1D.ravel('F')

# print(xi)
# print(eta)

dx_dxi = mesh1.dx_dxi(xi, eta)
dx_deta = mesh1.dx_deta(xi, eta)
dy_dxi = mesh1.dy_dxi(xi, eta)
dy_deta = mesh1.dy_deta(xi, eta)

g = (dx_dxi * dy_deta - dx_deta * dy_dxi)
# g11 = (dx_deta * dx_deta + dy_deta * dy_deta) / g
# g12 = -(dy_deta * dy_dxi + dx_dxi * dx_deta) / g
# g22 = (dx_dxi * dx_dxi + dy_dxi * dy_dxi) / g

# K11 = 1
# K12 = 0
# K22 = 1

# K11 = 5 / 20
# K12 = 0 / 20
# K22 = 4 / 20

K11 = 5 / 11
K12 = -3 / 11
K22 = 4 / 11

g11K = (dx_deta * dx_deta * K11 + 2 * dy_deta * dx_deta * K12 + dy_deta * dy_deta * K22) / g
g22K = (dx_dxi * dx_dxi * K11 + 2 * dy_dxi * dx_dxi * K12 + dy_dxi * dy_dxi * K22) / g
g12K = (dx_dxi * dx_deta * K11 + (dy_dxi * dx_deta + dx_dxi * dy_deta)
        * K12 + dy_dxi * dy_deta * K22) / g


# print(np.shape(g11))

M1_varun = mass_matrix_1_form_local(num_total_elements, pp[0], g11K, g12K, g22K)
# npt.assert_array_almost_equal(M1, M1_varun, decimal=15)


# g_11 = (dx_deta**2 * k_11 + 2 * dy_deta * dx_deta * k_12 + dy_deta**2 * k_22) / g
# g_12 = -(dx_dxi * dx_deta * k_11 + (dy_dxi * dx_deta + dx_dxi * dy_deta)
#          * k_12 + dy_dxi * dy_deta * k_22) / g
# g_22 = (dx_dxi**2 * k_11 + 2 * dy_dxi * dx_dxi * k_12 + dy_dxi**2 * k_22) / g

# print(np.shape(g), np.shape(g11), np.shape(g12), np.shape(g22))

# M1_2 = mass_matrix_1_form_local(pp, g11, g12, g22)

# print(time.time() - start_time)

M1_assembled = assemble(M1_varun, (fs_1_lobatto, fs_1_lobatto))

"""define the wedge product"""
E21 = d_21_lobatto_outer(pp)
W1 = basis_2.wedged(basis_0)

W1_E21 = np.dot(W1, E21)
W1_E21_local = np.repeat(W1_E21[:, :, np.newaxis], num_total_elements, axis=2)

W1_E21_assembled = assemble(W1_E21_local, (fs_0_gauss, fs_1_lobatto))

# print(time.time() - start_time)

"""assemble lhs"""
lhs = sparse.bmat([[M1_assembled, W1_E21_assembled.transpose()], [W1_E21_assembled, None]]).tolil()

# print(np.shape(lhs))
# A = np.linalg.det(lhs.todense())
# print(A)

"""assemble rhs"""
rhs1 = np.zeros(num_total_edges)[:, np.newaxis]

f_cochain_local = f_source.cochain_local[:, np.newaxis]
W1_f_local = np.tensordot(W1, f_cochain_local, axes=1)
# W1_f_local2 = np.zeros((num_local_surfaces, 1, num_total_elements))
# for i in range(num_total_elements):
#     W1_f_local2[:, :, i] = np.dot(W1, f_cochain_local[:, :, i])
# npt.assert_array_almost_equal(W1_f_local, W1_f_local2, decimal=15)

rhs2 = assemble_cochain2(W1_f_local, dof_map_lobatto_faces, num_total_faces)

rhs = np.vstack((rhs1, rhs2))

# print(time.time() - start_time)

"""implement boundary conditions"""

"""neuman boundary condition"""
"""dirichlet boundary condition"""

# bound_node_bottom, bound_node_top, bound_node_left, bound_node_right = ghost_nodes_dof_map_boundary()

"""solve linear system of equations"""

solution = sparse.linalg.spsolve(lhs.tocsc(), rhs)

end_time = time.time()

# print("The total time taken by the program is : ", end_time - start_time)

"""post processing / reconstruction"""
eta_plot = xi_plot = xi = eta = np.linspace(-1, 1, 30)

"""reconstruct fluxes"""

f_flux.cochain = solution[:fs_1_lobatto.num_dof]

f_flux.reconstruct(xi, eta)
(x_plot, y_plot), flux_x_plot, flux_y_plot = f_flux.export_to_plot()

flux_x_plot, flux_y_plot = flux_y_plot, flux_x_plot

# flux_x_exact_plot = 2 * np.pi * np.cos(2 * np.pi * x_plot) * np.sin(2 * np.pi * y_plot)
# flux_y_exact_plot = 2 * np.pi * np.sin(2 * np.pi * x_plot) * np.cos(2 * np.pi * y_plot)

# flux_x_exact_plot = 8 * np.pi * np.cos(2 * np.pi * x_plot) * np.sin(2 * np.pi * y_plot)
# flux_y_exact_plot = 10 * np.pi * np.sin(2 * np.pi * x_plot) * np.cos(2 * np.pi * y_plot)

flux_x_exact_plot = 8 * np.pi * np.cos(2 * np.pi * x_plot) * np.sin(
    2 * np.pi * y_plot) + 6 * np.pi * np.sin(2 * np.pi * x_plot) * np.cos(2 * np.pi * y_plot)
flux_y_exact_plot = 6 * np.pi * np.cos(2 * np.pi * x_plot) * np.sin(
    2 * np.pi * y_plot) + 10 * np.pi * np.sin(2 * np.pi * x_plot) * np.cos(2 * np.pi * y_plot)


"""reconstruct potential"""

f_phi.cochain = solution[fs_1_lobatto.num_dof:]
f_phi.reconstruct(xi, eta)

num_pts_y, num_pts_x = np.shape(f_phi.basis.xi)
num_el_x, num_el_y = f_phi.function_space.mesh.n_x, f_phi.function_space.mesh.n_y
x, y = f_phi.function_space.mesh.mapping(f_phi.basis.xi, f_phi.basis.eta)
# print(x[0, 0, :])
x_4d = x.reshape(num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
x = np.moveaxis(x_4d, 2, 1).reshape(
    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

y_4d = y.reshape(num_pts_y, num_pts_x,  num_el_y, num_el_x, order='F')
y = np.rollaxis(y_4d, 2, 1).reshape(
    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

recon_4d_dx = f_phi.reconstructed.reshape(num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
reconstructed_dx = np.moveaxis(recon_4d_dx, 2, 1).ravel('F').reshape(
    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

phi_exact_plot = np.sin(2 * np.pi * x_plot) * np.sin(2 * np.pi * y_plot)


def main():
    plt.figure(1)
    plt.contourf(x_plot, y_plot, flux_x_plot)
    plt.colorbar()

    plt.figure(2)
    plt.contourf(x_plot, y_plot, flux_x_exact_plot)
    plt.colorbar()

    print(np.max(flux_x_exact_plot), np.min(flux_x_exact_plot))
    print(np.max(flux_x_plot), np.min(flux_x_plot))

    plt.figure(3)
    plt.contourf(x_plot, y_plot, flux_y_plot)
    plt.colorbar()

    plt.figure(4)
    plt.contourf(x_plot, y_plot, flux_y_exact_plot)
    plt.colorbar()

    print(np.max(flux_y_exact_plot), np.min(flux_y_exact_plot))
    print(np.max(flux_y_plot), np.min(flux_y_plot))

    plt.figure(5)
    plt.contourf(x, y, reconstructed_dx)
    plt.colorbar()

    plt.figure(6)
    plt.contourf(x_plot, y_plot, phi_exact_plot)
    plt.colorbar()

    print(np.max(phi_exact_plot), np.min(phi_exact_plot))
    print(np.max(reconstructed_dx), np.min(reconstructed_dx))

    plt.show()


if __name__ == '__main__':
    main()
