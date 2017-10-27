import path_magic
import numpy as np
from mesh import CrazyMesh
from function_space import FunctionSpace
from basis_forms import BasisForm
from forms import Form
from inner_product import inner
from assemble import assemble
import scipy.sparse as spr
import scipy as sp
import matplotlib.pyplot as plt
from coboundaries import d


def pfun(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def ufun_u(x, y):
    return -np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)


def ufun_v(x, y):
    return np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)


def ffun(x, y):
    return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def single_element():
    mesh = CrazyMesh(2, (1, 1), ((-1, 1), (-1, 1)), curvature=0.1)
    p = 20, 20

    func_space_vort = FunctionSpace(mesh, '0-ext_gauss', (p[0] - 1, p[1] - 1), is_inner=False)
    func_space_outer_vel = FunctionSpace(
        mesh, '1-ext_gauss', (p[0] - 2, p[1] - 2), is_inner=False)
    func_space_inner_vel = FunctionSpace(mesh, '1-lobatto', p, is_inner=True)
    func_space_source = FunctionSpace(mesh, '2-lobatto', p, is_inner=True)

    basis_vort = BasisForm(func_space_vort)
    basis_vel_in = BasisForm(func_space_inner_vel)
    basis_vel_out = BasisForm(func_space_outer_vel)
    basis_2 = BasisForm(func_space_source)

    psi = Form(func_space_vort)
    u_in = Form(func_space_inner_vel)
    source = Form(func_space_source)
    source.discretize(ffun)

    M_1 = inner(basis_vel_in, basis_vel_in)
    W_11 = basis_vel_out.wedged(basis_vel_in)
    E_10_ext = d(func_space_vort)
    E_21_in = d(func_space_inner_vel)
    W_02 = basis_vort.wedged(basis_2)
    print("shapes : \n \
    M_1 : {0} \n \
    W_11 : {1} \n \
    E_10 : {2} \n \
    E_21_in : {3} \n \
    W_02 : {4} \n" .format(np.shape(M_1), np.shape(W_11), np.shape(E_10_ext), np.shape(E_21_in), np.shape(W_02)))
    # one element
    # print(func_space_inner_vel.num_dof, func_space_vort.num_dof)
    lhs_0 = np.hstack((M_1[:, :, 0], np.transpose(W_02 @ E_21_in)))
    col_size_0 = np.shape(lhs_0)[1]
    eW = W_02 @ E_21_in
    col_zeros = col_size_0 - np.shape(eW)[1]
    lhs_1 = np.hstack((eW, np.zeros(
        (func_space_source.num_dof, col_zeros))))
    lhs = np.vstack((lhs_0, lhs_1))

    rhs_source = (source.cochain @ W_02)[:, np.newaxis]
    rhs_zeros = np.zeros((np.shape(lhs)[0] - func_space_source.num_dof, 1))
    rhs = np.vstack((rhs_zeros, rhs_source))
    print(np.shape(lhs))
    solution = np.linalg.solve(lhs, rhs).flatten()
    print(np.shape(solution))
    print(func_space_vort.num_dof)
    u_in.cochain = solution[:func_space_inner_vel.num_dof]

    psi_zeros = np.zeros((func_space_vort.num_dof - func_space_vort.num_internal_local_dof))
    psi.cochain = np.append(solution[func_space_inner_vel.num_dof:],
                            np.zeros((func_space_vort.num_dof - func_space_vort.num_internal_local_dof)))
    xi = eta = np.linspace(-1, 1, 40)
    u_in.reconstruct(xi, eta)
    (x, y), u_x, u_y = u_in.export_to_plot()
    plt.contourf(x, y, u_x)
    plt.colorbar()
    plt.title("u_x inner")
    plt.show()
    psi.reconstruct(xi, eta)
    (x, y), psi_value = psi.export_to_plot()
    plt.contourf(x, y, psi_value)
    plt.title("psi outer"
              )
    plt.colorbar()
    plt.show()


def multiple_element():
    mesh = CrazyMesh(2, (5, 7), ((-1, 1), (-1, 1)), curvature=0.3)
    p = 10, 10
    func_space_vort = FunctionSpace(mesh, '0-ext_gauss', (p[0] - 1, p[1] - 1), is_inner=False)
    func_space_outer_vel = FunctionSpace(
        mesh, '1-total_ext_gauss', (p[0], p[1]), is_inner=False)
    # func_space_outer_vel = FunctionSpace(
    #     mesh, '1-gauss', (p[0], p[1]), is_inner=False)
    func_space_inner_vel = FunctionSpace(mesh, '1-lobatto', p, is_inner=True)
    func_space_inner_vel.dof_map.continous_dof = True
    func_space_source = FunctionSpace(mesh, '2-lobatto', p, is_inner=True)

    basis_vort = BasisForm(func_space_vort)
    basis_vel_in = BasisForm(func_space_inner_vel)
    basis_vel_in.quad_grid = 'lobatto'
    basis_vel_out = BasisForm(func_space_outer_vel)
    basis_vel_out.quad_grid = 'lobatto'
    basis_2 = BasisForm(func_space_source)

    psi = Form(func_space_vort)
    u_in = Form(func_space_inner_vel)
    source = Form(func_space_source)
    source.discretize(ffun)

    M_1 = inner(basis_vel_in, basis_vel_in)
    E_21_in = d(func_space_inner_vel)
    W_02 = basis_2.wedged(basis_vort)
    W_02_E21 = np.transpose(W_02 @ E_21_in)

    M_1 = assemble(M_1, (func_space_inner_vel, func_space_inner_vel))
    print(np.shape(M_1))
    W_02_E21 = assemble(W_02_E21, (func_space_inner_vel, func_space_vort))
    print(np.shape(W_02_E21))
    W_02 = assemble(W_02, (func_space_source, func_space_vort))

    lhs = spr.bmat([[M_1, W_02_E21], [W_02_E21.transpose(), None]])
    print(np.shape(lhs))
    rhs = np.zeros(np.shape(lhs)[0])
    rhs[-func_space_source.num_dof:] = W_02 @ source.cochain

    solution = spr.linalg.spsolve(lhs.tocsc(), rhs)
    u_in.cochain = solution[:func_space_inner_vel.num_dof]
    cochian_psi = np.zeros(func_space_vort.num_dof)
    cochian_psi[:func_space_vort.num_internal_dof] = solution[-func_space_vort.num_internal_dof:]
    psi.cochain = cochian_psi

    xi = eta = np.linspace(-1, 1, 40)
    u_in.reconstruct(xi, eta)
    (x, y), u_x, u_y = u_in.export_to_plot()
    plt.contourf(x, y, u_x)
    plt.colorbar()
    plt.title("u_x inner")
    plt.show()
    psi.reconstruct(xi, eta)
    (x, y), psi_value = psi.export_to_plot()
    plt.contourf(x, y, psi_value)
    plt.title("psi outer"
              )
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    multiple_element()
