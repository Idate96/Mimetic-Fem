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


def multiple_element():
    mesh = CrazyMesh(2, (1, 1), ((-1, 1), (-1, 1)), curvature=0.0)
    p = 10, 10
    func_space_vort = FunctionSpace(mesh, '0-lobatto', p, is_inner=False)
    func_space_outer_vel = FunctionSpace(
        mesh, '1-lobatto', p, is_inner=False)
    func_space_outer_vel.dof_map.continous_dof = True
    # func_space_outer_vel = FunctionSpace(
    #     mesh, '1-gauss', (p[0], p[1]), is_inner=False)
    func_space_inner_vel = FunctionSpace(
        mesh, '1-lobatto', (p[0] + 1, p[1] + 1), is_inner=True)
    func_space_inner_vel.dof_map.continous_dof = False
    func_space_source = FunctionSpace(mesh, '2-lobatto', (p[0] + 1, p[1] + 1), is_inner=True)

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
    # W_02 = basis_2.wedged(basis_vort)
    W_11 = basis_vel_out.wedged(basis_vel_in)
    E_10_out = d(func_space_vort)
    print(np.shape(W_11), np.shape(E_10_out))
    E_W = W_11 @ E_10_out
    print("shape ew : ", np.shape(E_W))

    M_1 = assemble(M_1, (func_space_inner_vel, func_space_inner_vel))
    print(func_space_source.num_local_dof)
    E_21_in = assemble(E_21_in, (func_space_source, func_space_inner_vel))
    E_W = assemble(E_W, (func_space_inner_vel, func_space_vort))

    # W_02 = assemble(W_02, (func_space_source, func_space_vort))

    lhs = spr.bmat([[M_1, E_W], [E_21_in, None]])
    rhs = np.zeros(np.shape(lhs)[0])
    rhs[-func_space_source.num_dof:] = source.cochain

    solution = spr.linalg.spsolve(lhs.tocsc(), rhs)
    u_in.cochain = solution[:func_space_inner_vel.num_dof]
    psi.cochain = solution[-func_space_vort.num_dof:]

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


def multiple_element_v1():
    mesh = CrazyMesh(2, (1, 1), ((-1, 1), (-1, 1)), curvature=0.0)
    p = 4, 4
    func_space_vort = FunctionSpace(mesh, '0-lobatto', p, is_inner=False)
    func_space_outer_vel = FunctionSpace(
        mesh, '1-lobatto', p, is_inner=False)
    func_space_outer_vel.dof_map.continous_dof = True
    # func_space_outer_vel = FunctionSpace(
    #     mesh, '1-gauss', (p[0], p[1]), is_inner=False)
    func_space_inner_vel = FunctionSpace(
        mesh, '1-gauss', (p[0], p[1]), is_inner=True)
    print('dof gauss :', func_space_inner_vel.num_dof)
    func_space_inner_vel.dof_map.continous_dof = False
    func_space_source = FunctionSpace(mesh, '2-gauss', (p[0] + 1, p[1] + 1), is_inner=True)
    print("dof source :", func_space_source.num_dof)

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
    # E_21_in = d(func_space_inner_vel)
    W_02 = basis_2.wedged(basis_vort)
    W_11 = basis_vel_in.wedged(basis_vel_out)
    E_10_out = d(func_space_vort)
    print(np.shape(W_11), np.shape(E_10_out))
    W_E = W_11 @ E_10_out
    W_11_inv = basis_vel_out.wedged(basis_vel_in)
    E_W = np.transpose(E_10_out) @ W_11_inv
    print("shape ew : ", np.shape(W_E))
    print(np.shape(W_02))

    M_1 = assemble(M_1, (func_space_inner_vel, func_space_inner_vel))
    print(func_space_source.num_local_dof)
    W_E = assemble(W_E, (func_space_outer_vel, func_space_vort))
    E_W = assemble(E_W, (func_space_vort, func_space_outer_vel))
    W_02 = assemble(W_02, (func_space_vort, func_space_source))

    lhs = spr.bmat([[M_1, W_E], [W_E.transpose(), None]])
    rhs = np.zeros(np.shape(lhs)[0])
    rhs[-func_space_source.num_dof:] = W_02 @ source.cochain

    solution = spr.linalg.spsolve(lhs.tocsc(), rhs)
    u_in.cochain = solution[:func_space_inner_vel.num_dof]
    psi.cochain = solution[-func_space_vort.num_dof:]

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
    plt.title("psi outer")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    multiple_element_v1()
