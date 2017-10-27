"""dual grid method"""

import time
import path_magic
from mesh import TransfiniteMesh, CrazyMesh
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
# def source(x, y): return -36 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 *
#  np.pi * y) + 24 * np.pi**2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

"""define anisotropic tensor"""


def k_11(x, y):
    alpha = 1e-4
    return (1e-3 * x**2 + y**2 + alpha) / (x**2 + y**2 + alpha)


def k_12(x, y):
    alpha = 1e-4
    return ((1e-3 - 1) * x * y) / (x**2 + y**2 + alpha)


def k_22(x, y):
    alpha = 1e-4
    return (x**2 + 1e-3 * y**2 + alpha) / (x**2 + y**2 + alpha)


def k11_dx(x, y):
    alpha = 1e-4
    return (2 * 1e-3 * x * (x**2 + y**2 + alpha) - 2 * x * (1e-3 * x**2 + y**2 + alpha)) / (x**2 + y**2 + alpha) ** 2


def k12_dx(x, y):
    alpha = 1e-4
    return ((x**2 + y**2 + alpha) * (1e-3 - 1) * y - 2 * (1e-3 - 1) * x**2 * y) / (x**2 + y**2 + alpha) ** 2


def k12_dy(x, y):
    alpha = 1e-4
    return ((x**2 + y**2 + alpha) * (1e-3 - 1) * x - 2 * (1e-3 - 1) * x * y**2) / (x**2 + y**2 + alpha) ** 2


def k22_dy(x, y):
    alpha = 1e-4
    return (2 * 0.001 * y * (x**2 + y**2 + alpha) - 2 * y * (x**2 + 0.001 * y**2 + alpha)) / (x**2 + y**2 + alpha) ** 2


"""define manufactured solution"""


def manufactured_solution(x, y): return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def phi_exact(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def d_phi_dx(x, y):
    return 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)


def d2_phi_dx2(x, y):
    return -4 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def d_phi_dy(x, y):
    return 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


def d2_phi_dy2(x, y):
    return -4 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def d2_phi_dxdy(x, y):
    return 4 * np.pi**2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)


"""define source"""


def source(x, y):
    return k11_dx(x, y) * d_phi_dx(x, y) + k_11(x, y) * d2_phi_dx2(x, y) + k12_dx(x, y) * d_phi_dy(x, y) + k_12(x, y) * d2_phi_dxdy(x, y) + k12_dy(x, y) * d_phi_dx(x, y) + k_12(x, y) * d2_phi_dxdy(x, y) + k22_dy(x, y) * d_phi_dy(x, y) + k_22(x, y) * d2_phi_dy2(x, y)


def flux_x_exact(x, y):
    return k_11(x, y) * d_phi_dx(x, y) + k_12(x, y) * d_phi_dy(x, y)
    # return 8 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) + 6 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


def flux_y_exact(x, y):
    return k_12(x, y) * d_phi_dx(x, y) + k_22(x, y) * d_phi_dy(x, y)
    # return 6 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) + 10 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


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


def main(el, poly_degree):
    dim = 2
    element_layout = (el + 1, el + 1)

    # print(element_layout)

    """define polynomial degree and inner/outer orientation"""

    pp = (poly_degree + 1, poly_degree + 1)                     # polynomial degree - primal mesh
    pd = (pp[0] - 1, pp[1] - 1)     # polynomial degree - dual mesh

    orientation_inner = True
    outer = False
    # is_inner = False        # orientation of primal mesh

    """define mesh"""

    bounds_domain = ((0, 1), (0, 1))
    curvature = 0.0
    mesh1 = CrazyMesh(dim, element_layout, bounds_domain, curvature)

    # gamma = (gamma1, gamma2, gamma3, gamma4)
    # dgamma = (dgamma1, dgamma2, dgamma3, dgamma4)
    # mesh1 = TransfiniteMesh(dim, element_layout, gamma, dgamma)

    """define function spaces used in problem"""

    fs_2_lobatto = FunctionSpace(mesh1, '2-lobatto', pp, outer)
    fs_1_lobatto = FunctionSpace(mesh1, '1-lobatto', pp, outer)
    fs_0_gauss = FunctionSpace(mesh1, '0-gauss', pd, inner)

    fs_1_lobatto.dof_map.continous_dof = True          # continuous elements

    """define forms and quad grid"""

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
    # anisotropic_tensor = MeshFunction(mesh1)
    # anisotropic_tensor.discrete_tensor = [
    #     k_11(element_layout), k_12(element_layout), k_22(element_layout)]
    # mesh function to inject the anisotropic tensor

    anisotropic_tensor = MeshFunction(mesh1)
    anisotropic_tensor.continous_tensor = [k_11, k_12, k_22]

    # mesh_k = MeshFunction(crazy_mesh)
    # mesh_k.continous_tensor = [diffusion_11, diffusion_12, diffusion_22]

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

    M1 = inner(basis_1, basis_1, anisotropic_tensor)
    M1_assembled = assemble(M1, (fs_1_lobatto, fs_1_lobatto))

    """define the wedge product"""
    E21 = d_21_lobatto_outer(pp)
    W1 = basis_2.wedged(basis_0)

    W1_E21 = np.dot(W1, E21)
    W1_E21_local = np.repeat(W1_E21[:, :, np.newaxis], num_total_elements, axis=2)

    W1_E21_assembled = assemble(W1_E21_local, (fs_0_gauss, fs_1_lobatto))

    """assemble lhs"""
    lhs = sparse.bmat([[M1_assembled, W1_E21_assembled.transpose()],
                       [W1_E21_assembled, None]]).tolil()

    # A = np.linalg.det(lhs.todense())
    # print(A)

    """assemble rhs"""
    rhs1 = np.zeros(num_total_edges)[:, np.newaxis]

    f_cochain_local = f_source.cochain_local[:, np.newaxis]
    W1_f_local = np.tensordot(W1, f_cochain_local, axes=1)
    rhs2 = assemble_cochain2(W1_f_local, dof_map_lobatto_faces, num_total_faces)

    rhs = np.vstack((rhs1, rhs2))

    # print(time.time() - start_time)

    """implement boundary conditions"""

    """neuman boundary condition"""
    """dirichlet boundary condition"""

    """solve linear system of equations"""

    solution = sparse.linalg.spsolve(lhs.tocsc(), rhs)

    end_time = time.time()

    print("The total time taken by the program is : ", end_time - start_time)

    """l2 error"""

    """post processing / reconstruction"""
    eta_plot = xi_plot = xi = eta = np.linspace(-1, 1, 30)

    """reconstruct fluxes"""

    f_flux.cochain = solution[:fs_1_lobatto.num_dof]

    f_flux.reconstruct(xi, eta)
    (x_plot, y_plot), flux_x_plot, flux_y_plot = f_flux.export_to_plot()

    flux_x_plot, flux_y_plot = flux_y_plot, flux_x_plot

    """reconstruct potential"""

    f_phi.cochain = solution[fs_1_lobatto.num_dof:]
    f_phi.reconstruct(xi, eta)

    (x_plot, y_plot), phi_plot = f_phi.export_to_plot()

    phi_exact_plot = np.sin(2 * np.pi * x_plot) * np.sin(2 * np.pi * y_plot)

    """l2 - error in (div u -f)"""

    div_u_sum = np.zeros(num_total_elements)

    for ele_num in range(num_total_elements):
        l2_div_u = np.dot(E21, f_flux.cochain_local[:, ele_num])[
            :, np.newaxis] - f_cochain_local[:, :, ele_num]
        div_u_sum[ele_num] = np.sum(l2_div_u)

    l2_err_div_u = np.linalg.norm(div_u_sum)
    l_inf_err_div_u = np.max(div_u_sum)

    """l2 - error in phi and flux """

    l2_err_phi = f_phi.l_2_norm(phi_exact)
    l2_err_flux = f_flux.l_2_norm((flux_y_exact, flux_x_exact))

    error = l2_err_phi, l2_err_flux, l2_err_div_u, l_inf_err_div_u

    print(l2_err_phi[0])
    print(l2_err_flux[0])

    # return error

    #
    plt.figure(1)
    plt.contourf(x_plot, y_plot, flux_x_plot)
    plt.colorbar()
    #
    # plt.figure(2)
    # plt.contourf(x_plot, y_plot, flux_x_exact_plot)
    # plt.colorbar()

    # print(np.max(flux_x_exact_plot), np.min(flux_x_exact_plot))
    # print(np.max(flux_x_plot), np.min(flux_x_plot))
    #
    plt.figure(3)
    plt.contourf(x_plot, y_plot, flux_y_plot)
    plt.colorbar()
    #
    # plt.figure(4)
    # plt.contourf(x_plot, y_plot, flux_y_exact_plot)
    # plt.colorbar()

    # print(np.max(flux_y_exact_plot), np.min(flux_y_exact_plot))
    # print(np.max(flux_y_plot), np.min(flux_y_plot))
    #
    plt.figure(5)
    plt.contourf(x_plot, y_plot, phi_plot)
    plt.colorbar()

    # plt.figure(6)
    # plt.contourf(x_plot, y_plot, phi_exact_plot)
    # plt.colorbar()

    # print(np.max(phi_exact_plot), np.min(phi_exact_plot))
    # print(np.max(phi_plot), np.min(phi_plot))

    plt.show()


if __name__ == '__main__':

    el = 10
    poly_degree = 10
    error = main(el, poly_degree)

    # num_elements = 10
    # num_poly_degree = 10
    # phi_err = np.zeros((num_elements, num_poly_degree))
    # flux_err = np.zeros((num_elements, num_poly_degree))
    # div_u_err = np.zeros((num_elements, num_poly_degree))
    # div_u_err_max = np.zeros((num_elements, num_poly_degree))
    # for el in range(num_elements):
    #     for poly_degree in range(num_poly_degree):
    #         error = main(el, poly_degree)
    #         phi_err[el, poly_degree] = error[0][0]
    #         flux_err[el, poly_degree] = error[1][0]
    #         div_u_err[el, poly_degree] = error[2]
    #         div_u_err_max[el, poly_degree] = error[3]
    #         print(el, poly_degree)
    # scipy.io.savemat(
    #     '/Users/varunjain/repos/mimeticfem/src/tests/tests_varun/portland_2017/data/rotating_anisotropy_phi_err_6x20.mat', mdict={'err_phi': phi_err})
    # scipy.io.savemat(
    #     '/Users/varunjain/repos/mimeticfem/src/tests/tests_varun/portland_2017/data/rotating_anisotropy_flux_err_6x20.mat', mdict={'err_flux': flux_err})
    # scipy.io.savemat(
    #     '/Users/varunjain/repos/mimeticfem/src/tests/tests_varun/portland_2017/data/rotating_anisotropy_div_u_err_6x20.mat', mdict={'err_div_u': div_u_err})
    # scipy.io.savemat(
    #     '/Users/varunjain/repos/mimeticfem/src/tests/tests_varun/portland_2017/data/rotating_anisotropy_div_u_err_max_6x20.mat', mdict={'err_div_u_max': div_u_err_max})

    # num_elements = 5
    # num_poly_degree = 5
    # phi_err = np.zeros((num_elements, num_poly_degree))
    # flux_err = np.zeros((num_elements, num_poly_degree))
    # div_u_err = np.zeros((num_elements, num_poly_degree))
    # div_u_err_max = np.zeros((num_elements, num_poly_degree))
    # for el in range(num_elements):
    #     for poly_degree in range(num_poly_degree):
    #         error = main(el, poly_degree)
    #         phi_err[el, poly_degree] = error[0][0]
    #         flux_err[el, poly_degree] = error[1][0]
    #         div_u_err[el, poly_degree] = error[2]
    #         div_u_err_max[el, poly_degree] = error[3]
    #         print(el, poly_degree)
    # scipy.io.savemat(
    #     '/Users/varunjain/repos/mimeticfem/src/tests/tests_varun/portland_2017/data/rotating_anisotropy_phi_err_25x10.mat', mdict={'err_phi': phi_err})
    # scipy.io.savemat(
    #     '/Users/varunjain/repos/mimeticfem/src/tests/tests_varun/portland_2017/data/rotating_anisotropy_flux_err_25x10.mat', mdict={'err_flux': flux_err})
    # scipy.io.savemat(
    #     '/Users/varunjain/repos/mimeticfem/src/tests/tests_varun/portland_2017/data/rotating_anisotropy_div_u_err_25x10.mat', mdict={'err_div_u': div_u_err})
    # scipy.io.savemat(
    #     '/Users/varunjain/repos/mimeticfem/src/tests/tests_varun/portland_2017/data/rotating_anisotropy_div_u_err_max_25x10.mat', mdict={'err_div_u_max': div_u_err_max})

# arr = numpy.arange(10)
# arr = arr.reshape((3, 3))  # 2d array of 3x3

# scipy.io.savemat('c:/tmp/arrdata.mat', mdict={'arr': arr})
