"""this file contain's varun's functions"""

import path_magic
import numpy as np
from quadrature import lobatto_quad
from polynomials import lagrange_basis, edge_basis


def assemble_cochain(local_cochain, dof_map_form, num_total_elements, num_total_faces):
    """added on 24 aug, 2017 by vj"""
    num_local_dof = np.shape(local_cochain)[0]
    num_total_dof = num_total_faces

    global_cochain = np.zeros((num_total_dof, 1))
    for ele_num in range(num_total_elements):
        for local_dof in range(num_local_dof):
            global_dof = dof_map_form[ele_num, local_dof]
            global_cochain[global_dof, 0] += local_cochain[local_dof, 0, ele_num]
    return global_cochain

# for ele_num, x in enumerate(dof_map_lobatto_faces):
#     print(ele_num, x)
#
# for (x, y), ele_num in np.ndenumerate(dof_map_lobatto_faces):
#     print ele_num, local_dof, global_dof

# for x, y in np.ndindex(dof_map_lobatto_faces.shape):
#     print(rows, columns)


"""assert"""
# W1_f_local2 = np.zeros((num_local_surfaces, 1, num_total_elements))
# for i in range(num_total_elements):
#     W1_f_local2[:, :, i] = np.dot(W1, f_cochain_local[:, :, i])
# npt.assert_array_almost_equal(W1_f_local, W1_f_local2, decimal=15)


def assemble_cochain2(local_cochain, dof_map_form, num_total_dof):
    """added on 25 aug, 2017 by vj"""

    # INPUT - local element wise cochain
    #         gathering matrix for the cochain
    #         total degrees of freedom of the form
    # OUTPUT  - column vector of assembled cochain

    global_cochain = np.zeros((num_total_dof, 1))
    for (ele_num, local_dof), global_dof in np.ndenumerate(dof_map_form):
        global_cochain[global_dof, 0] += local_cochain[local_dof, 0, ele_num]

    return global_cochain


def mass_matrix_1_form_local(num_total_elements, pp, g11, g12, g22):
    """added on 29th august, 2017 by vj"""
    lobatto_nodes, wp = lobatto_quad(pp)

    hp = lagrange_basis(lobatto_nodes, lobatto_nodes)
    ep = edge_basis(lobatto_nodes, lobatto_nodes)

    quad_pts_x = pp + 1
    quad_pts_y = pp + 1

    M1 = np.zeros((2 * pp * (pp + 1), 2 * pp * (pp + 1), num_total_elements))

    for ele_num, i, j, k, l, p, q in np.ndindex(num_total_elements, pp, pp + 1, pp, pp + 1, quad_pts_x, quad_pts_y):
        edgeij = i * (pp + 1) + j
        edgekl = k * (pp + 1) + l
        node_pq = p * (pp + 1) + q
        M1[edgeij, edgekl, ele_num] += g11[node_pq, ele_num] * ep[i, p] * \
            hp[j, q] * ep[k, p] * hp[l, q] * wp[p] * wp[q]

    for ele_num, i, j, k, l, p, q in np.ndindex(num_total_elements, pp, pp + 1, pp + 1, pp, quad_pts_x, quad_pts_y):
        edgeij = i * (pp + 1) + j
        edgekl = pp * (pp + 1) + k * pp + l
        node_pq = p * (pp + 1) + q
        M1[edgeij, edgekl, ele_num] += g12[node_pq, ele_num] * ep[i, p] * \
            hp[j, q] * hp[k, p] * ep[l, q] * wp[p] * wp[q]

    for ele_num, i, j, k, l, p, q in np.ndindex(num_total_elements, pp + 1, pp, pp, pp + 1, quad_pts_x, quad_pts_y):
        edgeij = pp * (pp + 1) + i * pp + j
        edgekl = k * (pp + 1) + l
        node_pq = p * (pp + 1) + q
        M1[edgeij, edgekl, ele_num] += g12[node_pq, ele_num] * hp[i, p] * \
            ep[j, q] * ep[k, p] * hp[l, q] * wp[p] * wp[q]

    for ele_num, i, j, k, l, p, q in np.ndindex(num_total_elements, pp + 1, pp, pp + 1, pp, quad_pts_x, quad_pts_y):
        edgeij = pp * (pp + 1) + i * pp + j
        edgekl = pp * (pp + 1) + k * pp + l
        node_pq = p * (pp + 1) + q
        M1[edgeij, edgekl, ele_num] += g22[node_pq, ele_num] * hp[i, p] * \
            ep[j, q] * hp[k, p] * ep[l, q] * wp[p] * wp[q]

    return M1

# def assemble_matrix(local_matrix, dof_map_test, dof_map_form):
#     num_total_test
#     num_total_cochain
#     assembled_matrix = np.zeros((num_total_test, num_total_cochain))
#
#     for ele_num in range(num_total_elements):
#         for local_test in range(num_local_test):
#             for local_cochain in range(num_local_cochain):
#                 assembled_matrix[global_test,
#                                  global_cochain] += local_matrix[local_test, local_cochain, ele_num]
#     return assembled_matrix
#
#
# def ghost_nodes_dof_map_boundary():
#     px = pp[0]
#     dof_ghost_points_boundary_bottom = np.zeros((px * element_layout[0]))
#     dof_ghost_points_boundary_top = np.zeros((px * element_layout[0]))
#     dof_ghost_points_boundary_left = np.zeros((px * element_layout[1]))
#     dof_ghost_points_boundary_right = np.zeros((px * element_layout[1]))
#
#     dof_map_ext_gauss_nodes = fs_0_gauss.dof_map.dof_map
#     print(dof_map_ext_gauss_nodes)
#     print(np.shape(dof_map_ext_gauss_nodes))
#
#     for i in range(element_layout[0]):
#         # bottom boundary
#         dof_ghost_points_boundary_bottom[i * px: (i + 1) * px] = \
#             dof_map_ext_gauss_nodes[i * element_layout[1], px**2 + 2 * px:px**2 + 3 * px]
#         # top boundary
#         dof_ghost_points_boundary_top[i * px: (i + 1) * px] = dof_map_ext_gauss_nodes[(
#             i + 1) * element_layout[1] - 1, px**2 + 3 * px:px**2 + 4 * px]
#
#     for j in range(elements_layout[1]):
#         # left boundary
#         dof_ghost_points_boundary_left[j * px:(j + 1) * px] = \
#             dof_map_ext_gauss_nodes[j, px**2: px**2 + px]
#         # right boundary
#         dof_ghost_points_boundary_right[j * px:(j + 1) * px] = dof_map_ext_gauss_nodes[(
#             elements_layout[0] - 1) * elements_layout[1] + j, px**2 + px: px**2 + 2 * px]
#
#     return dof_ghost_points_boundary_bottom, dof_ghost_points_boundary_top, \
#         dof_ghost_points_boundary_left, dof_ghost_points_boundary_right
