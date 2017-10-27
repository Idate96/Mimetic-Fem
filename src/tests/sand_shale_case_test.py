import path_magic
import numpy as np
from numpy import meshgrid
from mesh import TransfiniteMesh
from function_space import FunctionSpace
from inner_product import MeshFunction, inner
from forms import Form
from basis_forms import BasisForm
from quadrature import lobatto_quad
from scipy.sparse import coo_matrix
from scipy import sparse
import matplotlib.pyplot as plt
from assemble import assemble
from coboundaries import d
import time

start_time = time.time()

# mesh parameters
dim = 2
elements_layout = (20, 20)

# function space parameters
p = 3, 3
is_inner = False

# define mesh


def gamma1(s): return s, 0 * np.ones(np.shape(s))


def gamma2(t): return 1 * np.ones(np.shape(t)), t


def gamma3(s): return s, 1 * np.ones(np.shape(s))


def gamma4(t): return 0 * np.ones(np.shape(t)), t


def dgamma1(s): return 1 * np.ones(np.shape(s)), 0 * np.ones(np.shape(s))


def dgamma2(t): return 0 * np.ones(np.shape(t)), 1 * np.ones(np.shape(t))


def dgamma3(s): return 1 * np.ones(np.shape(s)), 0 * np.ones(np.shape(s))


def dgamma4(t): return 0 * np.ones(np.shape(t)), 1 * np.ones(np.shape(t))


gamma = (gamma1, gamma2, gamma3, gamma4)
dgamma = (dgamma1, dgamma2, dgamma3, dgamma4)

sand_shale_mesh = TransfiniteMesh(dim, elements_layout, gamma, dgamma)
# check by plotting

# xi = eta = np.linspace(-1, 1, 10)
#
# for i in range(elements_layout[0] * elements_layout[1]):
#     x_bound_1, y_bound_1 = sand_shale_mesh.mapping(xi, -1 * np.ones(np.shape(eta)), i)
#     x_bound_2, y_bound_2 = sand_shale_mesh.mapping(1, eta, i)
#     x_bound_3, y_bound_3 = sand_shale_mesh.mapping(xi, 1, i)
#     x_bound_4, y_bound_4 = sand_shale_mesh.mapping(-1, eta, i)
#
#     # print("this is gamma 1", x_bound_1, y_bound_1)
#     #
#     # print("this is gamma 3", x_bound_3, y_bound_3)
#
#     plt.plot(x_bound_1, y_bound_1, 'r')
#     plt.plot(x_bound_2, y_bound_2, 'b')
#     plt.plot(x_bound_3, y_bound_3, 'g')
#     plt.plot(x_bound_4, y_bound_4, 'k')
#
# plt.show()

# xi, w = lobatto_quad(p[0])
# eta, w = lobatto_quad(p[1])
# xi_plot, eta_plot = meshgrid(xi, eta)
#
# for i in range(elements_layout[0] * elements_layout[1]):
#     x_plot, y_plot = sand_shale_mesh.mapping(xi_plot, eta_plot, i)
#
#     plt.plot(x_plot, y_plot, 'r+')
# plt.axis([0, 2, 0, 2])
# plt.show()


# define function space

func_space_2_lobatto = FunctionSpace(sand_shale_mesh, '2-lobatto', p, is_inner)
func_space_1_lobatto = FunctionSpace(sand_shale_mesh, '1-lobatto', p, is_inner)
# continuous numbering
func_space_1_lobatto.dof_map.continous_dof = True

# define diffusion tensor


def k_11():
    A = [(1, 12), (1, 14), (1, 16), (2, 10), (2, 11), (2, 14), (2, 18), (2, 20),
         (3, 5), (3, 19), (4, 2), (4, 3), (4, 8), (4, 15), (4, 19), (5, 2), (5, 3),
         (5, 6), (5, 7), (5, 12), (5, 19), (6, 10), (6, 16), (6, 17), (7, 1), (7, 3),
         (7, 6), (7, 8), (7, 9), (8, 3), (8, 6), (8, 7), (8, 8), (8, 11), (8, 15), (8, 19),
         (9, 5), (9, 7), (9, 8), (9, 10), (9, 20), (10, 1), (10, 11), (10, 16), (11, 10),
         (11, 12), (11, 18), (12, 1), (12, 3), (12, 8), (12, 9), (12, 10), (12, 13),
         (13, 18), (13, 19), (14, 9), (14, 14), (14, 15), (14, 18), (15, 4), (15, 8),
         (15, 14), (16, 7), (16, 12), (16, 19), (17, 10), (17, 16), (17, 20), (18, 6),
         (19, 5), (19, 9), (19, 10), (19, 11), (20, 1), (20, 4), (20, 5), (20, 10),
         (20, 11), (20, 16), (20, 18)]

    row_idx, column_idx = zip(*A)
    row_idx = [value - 1 for value in row_idx]
    column_idx = [value - 1 for value in column_idx]
    data = 10**-6 * np.ones(np.size(row_idx))
    k_11 = coo_matrix((data, (row_idx, column_idx)), shape=(20, 20)).toarray().ravel('F')
    k_11[np.where(k_11 < 10**-12)] = 1
    return k_11.reshape(1, 400)


def k_12(): return np.zeros((1, 400))


def k_22():
    return k_11()


anisotropic_tensor = MeshFunction(sand_shale_mesh)
anisotropic_tensor.discrete_tensor = [k_11(), k_12(), k_22()]

# define source term


def source(x, y):
    return np.zeros(np.shape(x))


form_source = Form(func_space_2_lobatto)
form_source.discretize(source)

# define basis forms

basis_1 = BasisForm(func_space_1_lobatto)
basis_2 = BasisForm(func_space_2_lobatto)

basis_1.quad_grid = 'lobatto'
basis_2.quad_grid = 'lobatto'

# solution form
phi_2 = Form(func_space_2_lobatto)
phi_2.basis.quad_grid = 'lobatto'

q_1 = Form(func_space_1_lobatto)
q_1.basis.quad_grid = 'lobatto'

form_source = Form(func_space_2_lobatto)
form_source.discretize(source)

# define inner products

# M_1k = inner(basis_1, basis_1, anisotropic_tensor)
# M2_E21 = inner(basis_2, d(basis_1))
# M_2 = inner(basis_2, basis_2)
#
# print("this ?", np.shape(M2_E21))
#
# # define assembly of incidence and mass matrices
#
# M_1k_assembled = assemble(M_1k, func_space_1_lobatto)
# N_2 = assemble(N_2, (func_space_2_lobatto, func_space_1_lobatto))
# M_2 = assemble(M_2, func_space_2_lobatto)
#
# lhs = sparse.bmat([[M_1k, N_2], [N_2.transpose(), None]]).tocsc()
# rhs_source = (form_source.cochain @ M_2)[:, np.newaxis]
# rhs_zeros = np.zeros(lhs.shape[0] - np.size(rhs_source))[:, np.newaxis]
# rhs = np.vstack((rhs_zeros, rhs_source))
#
# solution = sparse.linalg.spsolve(lhs, rhs)
# phi_2.cochain = solution[-func_space_2_lobatto.num_dof:]
#
# # sample the solution
# xi = eta = np.linspace(-1, 1, 200)
# phi_2.reconstruct(xi, eta)
# (x, y), data = phi_2.export_to_plot()
#
#
# # M2_E21_assembled = assemble(M2_E21, (func_space_2_lobatto, func_space_1_lobatto))
# # M2_assembled = assemble(M2, func_space_2_lobatto)
# # # define lhs
# # print(np.shape(M_1k_assembled), np.shape(M2_E21_assembled.transpose()))
# # lhs = sparse.bmat([[M_1k_assembled, M2_E21_assembled.transpose()],
# #                    [M2_E21_assembled, None]]).tocsc()
# #
# # # define rhs
# #
# # rhs_source = (form_source.cochain @ M2_assembled)[:, np.newaxis]
# # rhs_zeros = np.zeros(lhs.shape[0] - np.size(rhs_source))[:, np.newaxis]
# # rhs = np.vstack((rhs_zeros, rhs_source))
#
# # define boundary condition
#
# bottom, top, left, right = func_space_1_lobatto.dof_map.dof_map_boundary
#
# lhs[bottom] = 0
# lhs[bottom, bottom] = 1
#
# # solution = sparse.linalg.spsolve(lhs, rhs)
# # phi_2.cochain = solution[-func_space_2_lobatto.num_dof:]
# #
# # # sample the solution
# # xi = eta = np.linspace(-1, 1, 200)
# # phi_2.reconstruct(xi, eta)
# # (x, y), data = phi_2.export_to_plot()
# # plt.contourf(x, y, data)
# # plt.show()
#
#
# # solve for unknowns


M_1k = inner(basis_1, basis_1, anisotropic_tensor)
N_2 = inner(basis_2, d(basis_1))
M_2 = inner(basis_2, basis_2)
# assemble inner products
M_1k = assemble(M_1k, func_space_1_lobatto)
N_2 = assemble(N_2, (func_space_2_lobatto, func_space_1_lobatto))
M_2 = assemble(M_2, func_space_2_lobatto)

lhs = sparse.bmat([[M_1k, N_2.transpose()], [N_2, None]]).tolil()
rhs_source = (form_source.cochain @ M_2)[:, np.newaxis]
rhs_zeros = np.zeros(lhs.shape[0] - np.size(rhs_source))[:, np.newaxis]
rhs = np.vstack((rhs_zeros, rhs_source))
# print(func_space_1_lobatto.dof_map.dof_map_boundary)
bottom, top, left, right = func_space_1_lobatto.dof_map.dof_map_boundary
# flux = 0
lhs[bottom] = 0
lhs[bottom, bottom] = 1
lhs[top] = 0
lhs[top, top] = 1
# phi = 1
rhs[left] = 1

solution = sparse.linalg.spsolve(lhs.tocsc(), rhs)

# print(np.shape(solution))

phi_2.cochain = solution[-func_space_2_lobatto.num_dof:]

# print(np.shape(phi_2.cochain))

q_1.cochain = solution[:-func_space_2_lobatto.num_dof]

print(" ------ %s seconds ------", (time.time() - start_time))

print(np.sum(q_1.cochain[left]))
print(np.sum(q_1.cochain[right]))

#
# print(np.shape(q_1.cochain))

# sample the solution
xi = eta = np.linspace(-1, 1, 2)
phi_2.reconstruct(xi, eta)
q_1.reconstruct(xi, eta)
t_0 = time.time()
(x, y), data = phi_2.export_to_plot()
t_1 = time.time()
plt.contourf(x, y, data)
t_2 = time.time()
print(t_2 - t_1, t_1 - t_0)
plt.colorbar()
plt.show()
