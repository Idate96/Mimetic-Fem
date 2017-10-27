from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import scipy.sparse as sparse
from function_space import FunctionSpace, NextSpace
from forms import Form, AbstractForm, cochain_to_global
from basis_forms import AbstractBasisForm, BasisForm
import scipy as sp
import sys
import numpy as np
import time


def fetch_incidence(function_space):
    """Fetch the function that return the adequate incidence matrix for the function space elements."""
    if function_space.is_inner:
        orientation = 'inner'
    else:
        orientation = 'outer'
    degree, form_name = function_space.form_type.split('-')
    func_name = 'd_' + str(int(degree) + 1) + degree + '_' + form_name + '_' + orientation
    return getattr(sys.modules[__name__], func_name)(function_space.p)


def parse_quadrature(basis):
    quad_type = basis.quad_grid
    p = [(np.size(basis._quad_nodes[i]) - 1) for i in range(2)]
    return quad_type


def d(state):
    """Exterior derivative operator.

    The function reppresent the discrete version of the exterior derivative:
    the coboundary operator.
    The function can act on:
        -FunctionSpace,
        -Form,
        -BasisForm.

    It returns respectively:
        -the incidence matrix,
        -a form of degree + 1, whose cochain is the result of matrix multiplication of the original cochain with the incidence matrix,
        -a tuple made by the basisform, of 1 degree higher, and the incidence matrix.
    """
    if isinstance(state, FunctionSpace):
        incidence_matrix = fetch_incidence(state)
        return incidence_matrix

    if isinstance(state, AbstractForm):
        function_space = state.function_space
        incidence_matrix = fetch_incidence(function_space)
        # generate the next func space in the de Rham sequence
        d_function_space = NextSpace(function_space)
        # calculate the new cochain
        d_form = Form(d_function_space)
        d_form.cochain_local = np.dot(incidence_matrix, state.cochain_local)
        d_form.cochain_to_global()
        return d_form

    if isinstance(state, AbstractBasisForm):
        function_space = state.function_space
        incidence_matrix = fetch_incidence(function_space)
        # generate the next func space in the de Rham sequence
        d_function_space = NextSpace(function_space)
        d_basis = BasisForm(d_function_space)
        d_basis.quad_grid = parse_quadrature(state)
        return (d_basis, incidence_matrix)


def d_10_lobatto_outer(p):
    px, py = p
    total_edges = px * (py + 1) + py * (px + 1)
    total_nodes = (px + 1) * (py + 1)
    E10 = np.zeros((total_edges, total_nodes))

    for i in range(px):
        for j in range(py + 1):
            edgeij = i * (py + 1) + j
            node_1 = i * (py + 1) + j
            node_2 = (i + 1) * (py + 1) + j

            E10[edgeij, node_1] = 1
            E10[edgeij, node_2] = -1

    for i in range(px + 1):
        for j in range(py):
            edgeij = px * (py + 1) + py * i + j
            node_1 = i * (py + 1) + j
            node_2 = i * (py + 1) + j + 1

            E10[edgeij, node_1] = -1
            E10[edgeij, node_2] = +1

    return E10


def d_10_lobatto_inner(p):
    px, py = p
    total_edges = px * (py + 1) + py * (px + 1)
    total_nodes = (px + 1) * (py + 1)
    E10 = np.zeros((total_edges, total_nodes))

    for i in range(px):
        for j in range(py + 1):
            edgeij = i * (py + 1) + j
            node_1 = i * (py + 1) + j
            node_2 = (i + 1) * (py + 1) + j

            E10[edgeij, node_1] = 1
            E10[edgeij, node_2] = -1

    for i in range(px + 1):
        for j in range(py):
            edgeij = px * (py + 1) + py * i + j
            node_1 = i * (py + 1) + j
            node_2 = i * (py + 1) + j + 1

            E10[edgeij, node_1] = 1
            E10[edgeij, node_2] = -1

    return -E10


def d_21_lobatto_inner(p):
    px, py = p
    total_vol = px * py
    total_edges = px * (py + 1) + py * (px + 1)
    E21 = np.zeros((total_vol, total_edges))

    for i in range(px):
        for j in range(py):
            volij = i * py + j
            edge_bottom = i * (py + 1) + j
            edge_top = i * (py + 1) + j + 1
            edge_left = (py + 1) * px + i * py + j
            edge_right = (py + 1) * px + (i + 1) * py + j

            E21[volij, edge_bottom] = 1
            E21[volij, edge_top] = -1
            E21[volij, edge_left] = -1
            E21[volij, edge_right] = 1
    return E21


def d_21_lobatto_outer(p):
    print("hello, THIS is d_21_lobatto_outer")
    px, py = p
    total_vol = px * py
    total_edges = px * (py + 1) + py * (px + 1)
    E21 = np.zeros((total_vol, total_edges))

    for i in range(px):
        for j in range(py):
            volij = i * py + j
            edge_bottom = i * (py + 1) + j
            edge_top = i * (py + 1) + j + 1
            edge_left = (py + 1) * px + i * py + j
            edge_right = (py + 1) * px + (i + 1) * py + j

            E21[volij, edge_bottom] = -1
            E21[volij, edge_top] = +1
            E21[volij, edge_left] = -1
            E21[volij, edge_right] = +1
    return E21


def d_10_ext_gauss_outer(p):
    px, py = p[0] + 1, p[1] + 1
    total_edges = px * (py + 1) + py * (px + 1)
    total_nodes = px * py + 2 * px + 2 * py
    internal_nodes = px * py
    E10 = np.zeros((total_edges, total_nodes))

    phi_left = range(internal_nodes, internal_nodes + py)
    phi_right = range(internal_nodes + py, internal_nodes + 2 * py)
    phi_bottom = range(internal_nodes + 2 * py, internal_nodes + 2 * py + px)
    phi_top = range(internal_nodes + 2 * py + px, internal_nodes + 2 * py + 2 * px)

    for i in range(px + 1):
        for j in range(py):
            edgeij = i * py + j
            if i == 0:
                node_1 = phi_left[j]
                node_2 = i * py + j
            elif i == px:
                node_1 = (i - 1) * py + j
                node_2 = phi_right[j]
            else:
                node_1 = (i - 1) * py + j
                node_2 = i * py + j

            E10[edgeij, node_1] = 1
            E10[edgeij, node_2] = -1

    for i in range(px):
        for j in range(py + 1):
            edgeij = (px + 1) * py + i * (py + 1) + j
            if j == 0:
                node_1 = phi_bottom[i]
                node_2 = i * py + j
            elif j == py:
                node_1 = i * py + j - 1
                node_2 = phi_top[i]
            else:
                node_1 = i * py + j - 1
                node_2 = i * py + j

            E10[edgeij, node_1] = -1
            E10[edgeij, node_2] = 1

    return E10

#
# def extended_gauss_E10(p):
#     px, py = p
#     total_edges = px * (py + 1) + py * (px + 1)
#     total_nodes = px * py + 2 * px + 2 * py
#     internal_nodes = px * py
#     E10 = np.zeros((total_edges, total_nodes))
#
#     phi_left = range(internal_nodes, internal_nodes + py)
#     phi_right = range(internal_nodes + py, internal_nodes + 2 * py)
#     phi_bottom = range(internal_nodes + 2 * py, internal_nodes + 2 * py + px)
#     phi_top = range(internal_nodes + 2 * py + px, internal_nodes + 2 * py + 2 * px)
#
#     for i in range(px):
#         for j in range(py + 1):
#             edgeij = (px + 1) * py + i * (py + 1) + j
#             if j == 0:
#                 E10[edgeij, phi_bottom[i]] = 1
#                 E10[edgeij, i * py + j] = -1
#             elif j == py:
#                 E10[edgeij, i * py + j - 1] = 1
#                 E10[edgeij, phi_top[i]] = -1
#             else:
#                 E10[edgeij, i * py + j - 1] = 1
#                 E10[edgeij, i * py + j] = -1
#
#     for i in range(px + 1):
#         for j in range(py):
#             edgeij = i * py + j
#             if i == 0:
#                 E10[edgeij, phi_left[j]] = 1
#                 E10[edgeij, i * py + j] = -1
#             elif i == px:
#                 E10[edgeij, (i - 1) * py + j] = 1
#                 E10[edgeij, phi_right[j]] = -1
#             else:
#                 E10[edgeij, (i - 1) * py + j] = 1
#                 E10[edgeij, i * py + j] = -1
#     return E10
#

# def d_10_lobatto_inner(p):
#    # raise UserWarning("Deprecated coboudnary")
#    p, py = p
#    # t0 = time.time()
#    main_seq = np.arange(p * (p + 1))
#
#    rows = np.zeros(4 * p * (p + 1))
#    rows[:p * (p + 1)] = rows[p * (p + 1):p * (p + 1) * 2] = main_seq
#    rows[p * (p + 1) * 2:p * (p + 1) * 3] = rows[p * (p + 1)
#                                                 * 3:] = main_seq + p * (p + 1)
#    # print(rows)
#    columns = np.zeros(4 * p * (p + 1))
#    columns[:p * (p + 1)] = main_seq
#    columns[p * (p + 1):2 * p * (p + 1)] = main_seq + (p + 1)
#    seq = np.arange(p)
#    for index in range(p + 1):
#        columns[2 * (p + 1) * p + p * index: 2 * (p + 1) * p +
#                p * (index + 1)] = seq + index * (p + 1)
#    columns[3 * (p + 1) * p:] = columns[2 * (p + 1) * p:3 * (p + 1) * p] + 1
#
#    data = np.ones(4 * p * (p + 1))
#    data[:p * (p + 1)] *= -1
#    data[p * (p + 1) * 2:p * (p + 1) * 3] *= -1
#    # print(data)
#    d_zero_sparse = sparse.coo_matrix(
#        (data, (rows, columns)), shape=(2 * p * (p + 1), (p + 1)**2))
#
#    return d_zero_sparse.toarray()


def d_10_ext_gauss_inner(p):
    px, py = p
    px += 1
    py += 1
    num_ghosts = 2 * (px + 1) + 2 * (py + 1)

    total_edges = px * (py + 1) + py * (px + 1)
    total_nodes = px * py + 2 * px + 2 * py
    internal_nodes = px * py
    E10 = np.zeros((total_edges, total_nodes))

    phi_left = range(internal_nodes, internal_nodes + py)
    phi_right = range(internal_nodes + py, internal_nodes + 2 * py)
    phi_bottom = range(internal_nodes + 2 * py, internal_nodes + 2 * py + px)
    phi_top = range(internal_nodes + 2 * py + px, internal_nodes + 2 * py + 2 * px)

    for i in range(px):
        for j in range(py + 1):
            edgeij = (px + 1) * py + i * (py + 1) + j
            if j == 0:
                E10[edgeij, phi_bottom[i]] = 1
                E10[edgeij, i * py + j] = -1
            elif j == py:
                E10[edgeij, i * py + j - 1] = 1
                E10[edgeij, phi_top[i]] = -1
            else:
                E10[edgeij, i * py + j - 1] = 1
                E10[edgeij, i * py + j] = -1

    for i in range(px + 1):
        for j in range(py):
            edgeij = i * py + j
            if i == 0:
                E10[edgeij, phi_left[j]] = 1
                E10[edgeij, i * py + j] = -1
            elif i == px:
                E10[edgeij, (i - 1) * py + j] = 1
                E10[edgeij, phi_right[j]] = -1
            else:
                E10[edgeij, (i - 1) * py + j] = 1
                E10[edgeij, i * py + j] = -1
    E_ghosts = np.zeros((num_ghosts, total_nodes))
    E_10_with_ghosts = np.vstack((E10, E_ghosts))
    return -E_10_with_ghosts

# def extended_gauss_E10(p):
#     px, py = p
#     total_edges = px * (py + 1) + py * (px + 1)
#     total_nodes = px * py + 2 * px + 2 * py
#     internal_nodes = px * py
#     E10 = np.zeros((total_edges, total_nodes))
#
#     phi_left = range(internal_nodes, internal_nodes + py)
#     phi_right = range(internal_nodes + py, internal_nodes + 2 * py)
#     phi_bottom = range(internal_nodes + 2 * py, internal_nodes + 2 * py + px)
#     phi_top = range(internal_nodes + 2 * py + px, internal_nodes + 2 * py + 2 * px)
#
#     for i in range(px):
#         for j in range(py + 1):
#             edgeij = (px + 1) * py + i * (py + 1) + j
#             if j == 0:
#                 E10[edgeij, phi_bottom[i]] = 1
#                 E10[edgeij, i * py + j] = -1
#             elif j == py:
#                 E10[edgeij, i * py + j - 1] = 1
#                 E10[edgeij, phi_top[i]] = -1
#             else:
#                 E10[edgeij, i * py + j - 1] = 1
#                 E10[edgeij, i * py + j] = -1
#
#     for i in range(px + 1):
#         for j in range(py):
#             edgeij = i * py + j
#             if i == 0:
#                 E10[edgeij, phi_left[j]] = 1
#                 E10[edgeij, i * py + j] = -1
#             elif i == px:
#                 E10[edgeij, (i - 1) * py + j] = 1
#                 E10[edgeij, phi_right[j]] = -1
#             else:
#                 E10[edgeij, (i - 1) * py + j] = 1
#                 E10[edgeij, i * py + j] = -1
#     return E10

#
# def d_10_lobatto_inner(p):
#     # raise UserWarning("Deprecated coboudnary")
#     p, py = p
#     # t0 = time.time()
#     main_seq = np.arange(p * (p + 1))
#
#     rows = np.zeros(4 * p * (p + 1))
#     rows[:p * (p + 1)] = rows[p * (p + 1):p * (p + 1) * 2] = main_seq
#     rows[p * (p + 1) * 2:p * (p + 1) * 3] = rows[p * (p + 1)
#                                                  * 3:] = main_seq + p * (p + 1)
#     # print(rows)
#     columns = np.zeros(4 * p * (p + 1))
#     columns[:p * (p + 1)] = main_seq
#     columns[p * (p + 1):2 * p * (p + 1)] = main_seq + (p + 1)
#     seq = np.arange(p)
#     for index in range(p + 1):
#         columns[2 * (p + 1) * p + p * index: 2 * (p + 1) * p +
#                 p * (index + 1)] = seq + index * (p + 1)
#     columns[3 * (p + 1) * p:] = columns[2 * (p + 1) * p:3 * (p + 1) * p] + 1
#
#     data = np.ones(4 * p * (p + 1))
#     data[:p * (p + 1)] *= -1
#     data[p * (p + 1) * 2:p * (p + 1) * 3] *= -1
#     # print(data)
#     d_zero_sparse = sparse.coo_matrix(
#         (data, (rows, columns)), shape=(2 * p * (p + 1), (p + 1)**2))
#
#     return d_zero_sparse.toarray()


def d_21_ext_gauss_outer(p):

    px, py = p
    px += 1
    py += 1

    internal_edges = (px + 1) * py + (py + 1) * px
    ghost_edges = (px + 1 + py + 1) * 2
    total_edges = internal_edges + ghost_edges
    total_volumes = (px + 1) * (py + 1)

    E21 = np.zeros((total_volumes, total_edges))

    edge_bottom = range(internal_edges, internal_edges + px + 1)
    edge_top = range(internal_edges + px + 1, internal_edges + 2 * px + 2 * 1)
    edge_left = range(internal_edges + 2 * px + 2 * 1, internal_edges + 2 * px + 2 * 1 + py + 1)
    edge_right = range(internal_edges + 2 * px + 2 * 1 + py + 1,
                       internal_edges + 2 * px + 2 * py + 4 * 1)

    for i in range(px + 1):
        for j in range(py + 1):
            volij = i * (py + 1) + j

            if i == 0:
                edgeij_left = edge_left[j]
            else:
                edgeij_left = (px + 1) * py + (i - 1) * (py + 1) + j

            if i == px:
                edgeij_right = edge_right[j]
            else:
                edgeij_right = (px + 1) * py + i * (py + 1) + j

            if j == 0:
                edgeij_bottom = edge_bottom[i]
            else:
                edgeij_bottom = i * py + (j - 1)

            if j == py:
                edgeij_top = edge_top[i]
            else:
                edgeij_top = i * py + j

            E21[volij, edgeij_bottom] = -1
            E21[volij, edgeij_top] = 1
            E21[volij, edgeij_left] = -1
            E21[volij, edgeij_right] = 1

    return E21


def d_21_ext_gauss_inner(p):
    print("hallo, world, this is inner ext_gauss E21")
    px, py = p
    px += 1
    py += 1
    internal_edges = (px + 1) * py + (py + 1) * px
    ghost_edges = (px + 1 + py + 1) * 2
    total_edges = internal_edges + ghost_edges
    total_volumes = (px + 1) * (py + 1)

    E21 = np.zeros((total_volumes, total_edges))

    edge_bottom = range(internal_edges, internal_edges + px + 1)
    edge_top = range(internal_edges + px + 1, internal_edges + 2 * px + 2 * 1)
    edge_left = range(internal_edges + 2 * px + 2 * 1, internal_edges + 2 * px + 2 * 1 + py + 1)
    edge_right = range(internal_edges + 2 * px + 2 * 1 + py + 1,
                       internal_edges + 2 * px + 2 * py + 4 * 1)

    for i in range(px + 1):
        for j in range(py + 1):
            volij = i * (py + 1) + j

            if i == 0:
                edgeij_left = edge_left[j]
            else:
                edgeij_left = (px + 1) * py + (i - 1) * (py + 1) + j

            if i == px:
                edgeij_right = edge_right[j]
            else:
                edgeij_right = (px + 1) * py + i * (py + 1) + j

            if j == 0:
                edgeij_bottom = edge_bottom[i]
            else:
                edgeij_bottom = i * py + (j - 1)

            if j == py:
                edgeij_top = edge_top[i]
            else:
                edgeij_top = i * py + j

            E21[volij, edgeij_bottom] = -1
            E21[volij, edgeij_top] = +1
            E21[volij, edgeij_left] = -1
            E21[volij, edgeij_right] = +1
    return E21


if __name__ == '__main__':
    p = (0, 0)
    E = d_21_ext_gauss_inner(p)
    print(E)
