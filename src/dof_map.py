"""This module contains the classes encapsulating the mapping between the local and the global dofs.
"""
import numpy as np
import sys
from mesh import CrazyMesh


class DofMap(object):
    """Computes the map between the local and the global degrees of freedom."""

    def __init__(self, function_space):
        self.mesh_type = type(function_space.mesh).__name__
        if isinstance(function_space.mesh.n, tuple):
            # structured grid
            self.n = (function_space.mesh.n_x, function_space.mesh.n_y)
        self.form_type = function_space.form_type
        self.p = function_space.p
        self._dof_map = None
        self._dof_map_internal = None
        # TODO: 0 and 2 flexible not supported
        self.form_to_dof = {'0-lobatto': ['lobatto_nodes_discontinous', 'lobatto_nodes_continous'],
                            '0-gauss': 'gauss_nodes',
                            '0-ext_gauss': 'ext_gauss_nodes',
                            '1-lobatto': ['lobatto_edges_discontinous', 'lobatto_edges_continous'],
                            '1-gauss': 'gauss_edges',
                            '1-ext_gauss': 'ext_gauss_edges',
                            '1-total_ext_gauss': 'lobatto_edges_continous',
                            '2-lobatto': 'lobatto_faces',
                            '2-gauss': 'gauss_faces',
                            '2-ext_gauss': 'ext_gauss_faces'}
        self.continous_dof = False
        self.mesh_to_dof = {'CrazyMesh': 'crazy', 'TransfiniteMesh': 'crazy'}
        self._interface_edges = None
        self._interface_nodes = None
        self._dof_map_boundary = None

    @property
    def dof_map(self):
        """Return the map from the local to the global dof."""
        if self._dof_map is None:
            self._dof_map = self.fetch_map_dof()
#            print(self._dof_map)
        return self._dof_map

    @dof_map.setter
    def dof_map(self, dof_map):
        self._dof_map = dof_map

    def fetch_map_dof(self):
        """Fetch the mapping for the dof."""
        mesh_str = self.mesh_to_dof[self.mesh_type]
        dof_str = self.form_to_dof[self.form_type]
        if '0-lobatto' in self.form_type or '1-lobatto' in self.form_type:
            dof_str = dof_str[self.continous_dof]
        dof_map = getattr(sys.modules[__name__], 'dof_map_' +
                          mesh_str + '_' + dof_str)(self.n, self.p)
#        print(dof_map)
        return dof_map

    @property
    def dof_map_internal(self):
        """Return the map from the local to the global dof."""
        if self._dof_map_internal is None:
            self._dof_map_internal = self.fetch_map_dof_internal()
#            print(self._dof_map)
        return self._dof_map_internal

    @dof_map_internal.setter
    def dof_map_internal(self, dof_map_internal):
        self._dof_map_internal = dof_map_internal

    def fetch_map_dof_internal(self):
        """Fetch the mapping for the dof."""
        mesh_str = self.mesh_to_dof[self.mesh_type]
        dof_str = self.form_to_dof[self.form_type]
        # TODO: support px and py
        dof_map_internal = getattr(sys.modules[__name__], 'dof_map_' +
                                   mesh_str + '_' + dof_str + '_internal')(self.n, self.p)
#        print(dof_map)
        return dof_map_internal

    @property
    def interface_edges(self, interface_edges):
        """Return edges numbering """
        if self._interface_edges is None:
            self._interface_edges = self.fetch_interface_edges()
#            print(self._dof_map)
        return self._interface_edges

    def fetch_interface_edges(self):
        """Fetch the mapping for the dof."""
        mesh_str = self.mesh_to_dof[self.mesh_type]
        dof_str = self.form_to_dof[self.form_type]
        interface_edges = getattr(
            sys.modules[__name__], 'dof_map_' + mesh_str + '_' + dof_str + '_interface')(self.n, self.p)
#        print(dof_map)
        return interface_edges

    @property
    def interface_nodes(self, interface_nodes):
        """Return edges numbering """
        if self._interface_nodes is None:
            self._interface_nodes = self.fetch_interface_nodes()
#            print(self._dof_map)
        return self._interface_nodes

    def fetch_interface_nodes(self):
        """Fetch the mapping for the dof."""
        mesh_str = self.mesh_to_dof[self.mesh_type]
        dof_str = self.form_to_dof[self.form_type]
        # TODO: support px and py
        interface_nodes = getattr(sys.modules[__name__], 'dof_map_' +
                                  mesh_str + '_' + dof_str + '_interface')(self.n, self.p)
#        print(dof_map)
        return interface_nodes

    @property
    def dof_map_boundary(self):
        """Return the map from the local to the global dof."""
        if self._dof_map_boundary is None:
            self._dof_map_boundary = self.fetch_map_dof_boundary()
#            print(self._dof_map)
        return self._dof_map_boundary

    @dof_map_boundary.setter
    def dof_map_boundary(self, dof_map_boundary):
        self._dof_map_boundary = dof_map_boundary

    def fetch_map_dof_boundary(self):
        """Fetch the mapping for the dof."""
        mesh_str = self.mesh_to_dof[self.mesh_type]
        dof_str = self.form_to_dof[self.form_type]
        if '0-lobatto' in self.form_type or '1-lobatto' in self.form_type:
            dof_str = dof_str[self.continous_dof]
        dof_map_boundary = getattr(sys.modules[__name__], 'dof_map_' +
                                   mesh_str + '_' + dof_str + '_boundary')(self.n, self.p)
        return dof_map_boundary

    #
    # def crazy_numbering_nodes(self):
    #     ny, nx = self.mesh.n_x, self.mesh.n_y
    #     global_indeces = np.rot90(np.arange((1 + (ny * self.p)) * (nx * self.p + 1)
    #                                         ).reshape(1 + (ny * self.p), (nx * self.p + 1)))
    #     self.global_numbering = np.zeros((nx * ny, (self.p + 1)**2))
    #     el_count = 0
    #     for j in range(ny):
    #         for i in range(nx - 1, -1, -1):
    #             self.global_numbering[el_count, :] = np.rot90(global_indeces[i * (self.p + 1) - i: (i + 1) * self.p + 1,
    #                                                                          j * (self.p + 1) - j: (j + 1) * self.p + 1], 3).reshape((self.p + 1)**2)
    #             el_count += 1
    #
    # def crazy_numbering_edges(self):
    #     ny, nx = self.mesh.n_x, self.mesh.n_y
    #     print(ny, nx, self.p)
    #     global_indeces_x = np.rot90(
    #         np.arange((ny * self.p) * (nx * self.p + 1)).reshape(ny * self.p, nx * self.p + 1), 1)
    #     global_indeces_y = np.rot90(np.arange(
    #         (ny * self.p + 1) * (nx * self.p)).reshape((ny * self.p + 1), (nx * self.p))) + np.max(global_indeces_x) + 1
    #     n_edges_el = self.p * (self.p + 1) * 2
    #     assert isinstance(n_edges_el, int)
    #     self.global_numbering = np.zeros((nx * ny, n_edges_el))
    #     el_count = 0
    #     for j in range(ny):
    #         for i in range(nx - 1, -1, -1):
    #             self.global_numbering[el_count, : int(n_edges_el / 2)] = \
    #                 np.rot90(global_indeces_x[i * (self.p + 1) - i:(i + 1) * (self.p + 1) - i,
    #                                           j * self.p:(j + 1) * self.p], 3).reshape(self.p * (self.p + 1))
    #             self.global_numbering[el_count, int(n_edges_el / 2):] = \
    #                 np.rot90(global_indeces_y[i * (self.p):(i + 1) * (self.p), j * (self.p + 1) -
    #                                           j: (j + 1) * (self.p + 1) - j], 3).reshape(self.p * (self.p + 1))
    #             el_count += 1
    #
    # def crazy_numbering_faces(self):
    #     ny, nx = self.mesh.n_x, self.mesh.n_y
    #     global_indeces = np.rot90(
    #         np.arange((self.p * nx) * (self.p * ny)).reshape(ny * self.p, nx * self.p))
    #     self.global_numbering = np.zeros((nx * ny, self.p**2))
    #     el_count = 0
    #     for j in range(ny):
    #         for i in range(nx - 1, -1, -1):
    #             self.global_numbering[el_count, :] = np.rot90(global_indeces[i *
    #                                                                          self.p:(i + 1) * self.p, j * self.p:(j + 1) * self.p], 3).reshape(self.p**2)
    #             el_count += 1


def dof_map_crazy_lobatto_nodes_continous_boundary(n, p):
    px, py = p
    p = px
    ny, nx = n
    global_indeces = np.rot90(np.arange((1 + (ny * p)) * (nx * p + 1)
                                        ).reshape(1 + (ny * p), (nx * p + 1)))
    bottom_nodes = global_indeces[-1, :]
    top_nodes = global_indeces[0, :]
    left_nodes = global_indeces[:, 0]
    right_nodes = global_indeces[:, -1]
    return bottom_nodes, top_nodes, left_nodes[::-1], right_nodes[::-1]


def dof_map_crazy_lobatto_edges_continous_boundary(n, p):
    px, py = p
    p = px
    ny, nx = n
    global_indeces_x = np.rot90(
        np.arange((ny * p) * (nx * p + 1)).reshape(ny * p, nx * p + 1), 1)

    bottom_nodes = global_indeces_x[-1, :]
    top_nodes = global_indeces_x[0, :]

    global_indeces_y = np.rot90(np.arange(
        (ny * p + 1) * (nx * p)).reshape((ny * p + 1), (nx * p))) + np.max(global_indeces_x) + 1

    left_nodes = global_indeces_y[:, 0]
    right_nodes = global_indeces_y[:, -1]

    return bottom_nodes, top_nodes, left_nodes[::-1], right_nodes[::-1]


def dof_map_crazy_lobatto_faces_boundary(n, p):
    px, py = p
    p = px
    ny, nx = n
    global_indeces = np.rot90(
        np.arange((p * nx) * (p * ny)).reshape(ny * p, nx * p))

    bottom_nodes = global_indeces[-1, :]
    top_nodes = global_indeces[0, :]
    left_nodes = global_indeces[:, 0]
    right_nodes = global_indeces[:, -1]
    return bottom_nodes, top_nodes, left_nodes[::-1], right_nodes[::-1]


def dof_map_crazy_lobatto_nodes_continous(n, p):
    """
    CrazyMesh, general numbering rule, standard lobatto 0-form

    """
    px, py = p
    p = px
    ny, nx = n
    global_indeces = np.rot90(np.arange((1 + (ny * p)) * (nx * p + 1)
                                        ).reshape(1 + (ny * p), (nx * p + 1)))
    global_numbering = np.zeros((nx * ny, (p + 1)**2), dtype=np.int64)
    el_count = 0
    for j in range(ny):
        for i in range(nx - 1, -1, -1):
            global_numbering[el_count, :] = np.rot90(
                global_indeces[i * (p + 1) - i: (i + 1) * p + 1, j * (p + 1) - j: (j + 1) * p + 1], 3).reshape((p + 1)**2)
            el_count += 1

    return global_numbering


def dof_map_crazy_lobatto_nodes_discontinous(n, p):
    """
    CrazyMesh, general numbering rule, standard lobatto 0-form

    """
#    px, py = p
#    ny, nx = n
#
#    N = (px + 1) * (py + 1)
#    global_numbering = np.zeros((nx * ny, N), dtype=np.int32)
#
#    local_numbering = np.array([int(i) for i in range(N)])
#
#    for i in range(nx):
#        for j in range(ny):
#            s = j + i * ny
#            global_numbering[s, :] = local_numbering + s * N
#
#    return global_numbering
    print("hello, world asdasdasfe")
    px, py = p
    p = px
    ny, nx = n
    global_indeces = np.rot90(np.arange((1 + (ny * p)) * (nx * p + 1)
                                        ).reshape(1 + (ny * p), (nx * p + 1)))
    global_numbering = np.zeros((nx * ny, (p + 1)**2), dtype=np.int64)
    el_count = 0
    for j in range(ny):
        for i in range(nx - 1, -1, -1):
            global_numbering[el_count, :] = np.rot90(
                global_indeces[i * (p + 1) - i: (i + 1) * p + 1, j * (p + 1) - j: (j + 1) * p + 1], 3).reshape((p + 1)**2)
            el_count += 1

    return global_numbering

    


#
# def dof_map_crazy_lobatto_edges(mesh, p):
#     """
#     CrazyMesh, general numbering rule, standard lobatto 1-form
#
#     """
#     ny, nx = mesh.n_x, mesh.n_y
#     global_indeces_x = np.rot90(
#         np.arange((ny * p) * (nx * p + 1)).reshape(ny * p, nx * p + 1), 1)
#     global_indeces_y = np.rot90(np.arange(
#         (ny * p + 1) * (nx * p)).reshape((ny * p + 1), (nx * p))) + np.max(global_indeces_x) + 1
#     n_edges_el = p * (p + 1) * 2
#     assert isinstance(n_edges_el, int)
#     global_numbering = np.zeros((nx * ny, n_edges_el), dtype=np.int64)
#     el_count = 0
#     for j in range(ny):
#         for i in range(nx - 1, -1, -1):
#             global_numbering[el_count, : int(n_edges_el / 2)] = \
#                 np.rot90(global_indeces_x[i * (p + 1) - i:(i + 1) * (p + 1) - i,
#                                           j * p:(j + 1) * p], 3).reshape(p * (p + 1))
#             global_numbering[el_count, int(n_edges_el / 2):] = \
#                 np.rot90(global_indeces_y[i * (p):(i + 1) * (p), j * (p + 1) -
#                                           j: (j + 1) * (p + 1) - j], 3).reshape(p * (p + 1))
#             el_count += 1
#     return global_numbering


def dof_map_crazy_lobatto_edges_continous(n, p):
    """
    CrazyMesh, general numbering rule, standard lobatto 1-form

    """
    ny, nx = n
    px, py = p
    p = px
    global_indeces_x = np.rot90(
        np.arange((ny * p) * (nx * p + 1)).reshape(ny * p, nx * p + 1), 1)
    global_indeces_y = np.rot90(np.arange(
        (ny * p + 1) * (nx * p)).reshape((ny * p + 1), (nx * p))) + np.max(global_indeces_x) + 1
    n_edges_el = p * (p + 1) * 2
    assert isinstance(n_edges_el, int)
    global_numbering = np.zeros((nx * ny, n_edges_el), dtype=np.int64)
    el_count = 0
    for j in range(ny):
        for i in range(nx - 1, -1, -1):
            global_numbering[el_count, : int(n_edges_el / 2)] = \
                np.rot90(global_indeces_x[i * (p + 1) - i:(i + 1) * (p + 1) - i,
                                          j * p:(j + 1) * p], 3).reshape(p * (p + 1))
            global_numbering[el_count, int(n_edges_el / 2):] = \
                np.rot90(global_indeces_y[i * (p):(i + 1) * (p), j * (p + 1) -
                                          j: (j + 1) * (p + 1) - j], 3).reshape(p * (p + 1))
            el_count += 1
    return global_numbering


def dof_map_crazy_lobatto_edges_interface(n, p):
    px, py = p
    p = px
    nx, ny = n
    global_numbering = np.zeros((nx * ny, 2 * p * (p + 1)), dtype=np.int32)
    local_numbering = np.array([int(i) for i in range(2 * p * (p + 1))])
    for i in range(nx):
        for j in range(ny):
            s = j + i * ny
            global_numbering[s, :] = local_numbering + 2 * p * (p + 1) * s
    interface_edge_pair = np.zeros((((nx - 1) * ny + nx * (ny - 1)) * p, 2), dtype=np.int32)
    n = 0
    for i in range(nx - 1):
        for j in range(ny):
            s1 = j + i * ny
            s2 = j + (i + 1) * ny
            for m in range(p):
                interface_edge_pair[n, 0] = global_numbering[s1, p * (p + 1) + p**2 + m]
                interface_edge_pair[n, 1] = global_numbering[s2, p * (p + 1) + m]
                n += 1
    for i in range(nx):
        for j in range(ny - 1):
            s1 = j + i * ny
            s2 = j + 1 + i * ny
            for m in range(p):
                interface_edge_pair[n, 0] = global_numbering[s1, (m + 1) * (p + 1) - 1]
                interface_edge_pair[n, 1] = global_numbering[s2,  m * (p + 1)]
                n += 1
    return interface_edge_pair


def dof_map_crazy_lobatto_edges_discontinous(n, p):
    px, py = p
    p = px
    nx, ny = n
    global_numbering = np.zeros((nx * ny, 2 * p * (p + 1)), dtype=np.int32)
    local_numbering = np.array([int(i) for i in range(2 * p * (p + 1))])

    for i in range(nx):
        for j in range(ny):
            s = j + i * ny
            global_numbering[s, :] = local_numbering + 2 * p * (p + 1) * s
    interface_edge_pair = np.zeros((((nx - 1) * ny + nx * (ny - 1)) * p, 2), dtype=np.int32)
    n = 0
    for i in range(nx - 1):
        for j in range(ny):
            s1 = j + i * ny
            s2 = j + (i + 1) * ny
            for m in range(p):
                interface_edge_pair[n, 0] = global_numbering[s1, p * (p + 1) + p**2 + m]
                interface_edge_pair[n, 1] = global_numbering[s2, p * (p + 1) + m]
                n += 1
    for i in range(nx):
        for j in range(ny - 1):
            s1 = j + i * ny
            s2 = j + 1 + i * ny
            for m in range(p):
                interface_edge_pair[n, 0] = global_numbering[s1, (m + 1) * (p + 1) - 1]
                interface_edge_pair[n, 1] = global_numbering[s2,  m * (p + 1)]
                n += 1
    return global_numbering


def dof_map_crazy_lobatto_faces(n, p):
    """
    CrazyMesh, general numbering rule, standard lobatto 1-form

    """
    px, py = p
    p = px
    ny, nx = n
    global_indeces = np.rot90(
        np.arange((p * nx) * (p * ny)).reshape(ny * p, nx * p))
    global_numbering = np.zeros((nx * ny, p**2), dtype=np.int64)
    el_count = 0
    for j in range(ny):
        for i in range(nx - 1, -1, -1):
            global_numbering[el_count, :] = np.rot90(global_indeces[i *
                                                                    p:(i + 1) * p, j * p:(j + 1) * p], 3).reshape(p**2)
            el_count += 1
    return global_numbering


def dof_map_crazy_gauss_nodes(n, p):
    """
    CrazyMesh, general numbering rule, standard gauss 0-form

    """
    p_x, p_y = p
    p = p_x + 1
    nx, ny = n
    Nx = nx * p
    Ny = ny * p
    global_numbering_M = np.array([int(i) for i in range(Nx * Ny)])
    global_numbering_M = np.reshape(global_numbering_M, (Ny, Nx), order='F')

    for j in range(nx):
        for i in range(ny):
            m = i * p
            n = j * p
            global_numbering_local = global_numbering_M[m: m + p, n: n + p]
            global_numbering_local = np.reshape(global_numbering_local, (1, p**2), 'F')

            if i == 0 and j == 0:
                global_numbering = global_numbering_local
            else:
                global_numbering = np.vstack((global_numbering, global_numbering_local))
    return global_numbering


def dof_map_crazy_gauss_edges(n, p):
    """
    CrazyMesh, general numbering rule, standard gauss 1-form

    """
    p_x, p_y = p
    p = p_x + 1
    nx, ny = n
    H = np.reshape(np.array([int(i) for i in range(nx * ny * (p - 1) * p)]),
                   (ny * p, nx * (p - 1)), 'F')
    V = np.reshape(np.array([int(i) for i in range(nx * ny * (p - 1) * p)]),
                   (ny * (p - 1), nx * p), 'F') + nx * ny * (p - 1) * p
    global_numbering = np.zeros(shape=(nx * ny, 2 * p * (p - 1)), dtype=np.int64)
    for j in range(nx):
        for i in range(ny):
            s = i + j * ny
            global_numbering[s, 0:p * (p - 1)] = np.reshape(H[i * p:(i + 1)
                                                              * p, j * (p - 1):(j + 1) * (p - 1)],  (1, p * (p - 1)), 'F')
            global_numbering[s, p * (p - 1): 2 * p * (p - 1)] = np.reshape(V[i *
                                                                             (p - 1):(i + 1) * (p - 1), j * p:(j + 1) * p],  (1, p * (p - 1)), 'F')
    return global_numbering


def dof_map_crazy_gauss_faces(n, p):
    """
    CrazyMesh, general numbering rule, standard gauss 2-form

    """
    global_numbering = dof_map_crazy_lobatto_faces(n, p)
    return global_numbering


def dof_map_crazy_ext_gauss_nodes(n, p):
    """
    CrazyMesh, general numbering rule, standard extended gauss 0-form

    """
    px, py = p
    p = px
    p += 1
    nx, ny = n
    global_numbering = np.zeros(shape=(nx * ny, (p + 2)**2 - 4), dtype=np.int64)
    global_numbering[:, 0: p**2] = dof_map_crazy_gauss_nodes(n, (p - 1, p - 1))
    for I in range(nx):
        for J in range(ny):
            eleid = J + I * ny

            global_numbering[eleid, p**2: p**2 + 2 * p] = np.array([int(i) for i in range(
                p**2 * nx * ny + J * (nx + 1) * p + I * p, p**2 * nx * ny + J * (nx + 1) * p + I * p + 2 * p, 1)])

            global_numbering[eleid, p**2 + 2 * p: p**2 + 4 * p] = np.array([int(i) for i in range(p**2 * nx * ny + p * nx * ny + ny * p + I * (
                ny + 1) * p + J * p, p**2 * nx * ny + p * nx * ny + ny * p + I * (ny + 1) * p + J * p + 2 * p, 1)])
    return global_numbering


def dof_map_crazy_ext_gauss_edges(n, p):
    """
    CrazyMesh, general numbering rule, standard extended gauss 1-form

    """
    px, py = p
    p = px
    p += 1
    nx, ny = n
    global_numbering = np.zeros(shape=(nx * ny, 2 * p * (p + 1) + 4 * (p + 1)), dtype=np.int32)
    global_numbering_internal = np.zeros(shape=(nx * ny, 2 * p * (p + 1)), dtype=np.int32)

    A = np.array([int(i) for i in range(2 * p * (p + 1))])
    for s in range(nx * ny):
        global_numbering[s, 0: 2 * p * (p + 1)] = s * 2 * p * (p + 1) + A
        global_numbering_internal[s, 0: 2 * p * (p + 1)] = s * 2 * p * (p + 1) + A

    M = 2 * p * (p + 1) * nx * ny
    N = M + (p + 1) * nx * (ny + 1)
    O = N + (p + 1) * ny * (nx + 1)
    V = np.array([int(i) for i in range(N, O, 1)]).reshape(((nx + 1),  (p + 1) * ny)).T
    P = 2 * p * (p + 1)
    for j in range(nx):
        for i in range(ny):
            s = i + j * ny
            global_numbering[s, P: P + 2 * (p + 1)] = np.array([int(i)
                                                                for i in range(M, M + 2 * (p + 1))])
            if (i + 1) % ny == 0:
                M += 2 * (p + 1)
            else:
                M += (p + 1)
            global_numbering[s, P + 2 * (p + 1): P + 4 * (p + 1)] = np.reshape(
                V[i * (p + 1): (i + 1) * (p + 1), j: j + 2], (1, 2 * (p + 1)), 'F')
    return global_numbering


def dof_map_crazy_ext_gauss_nodes_internal(n, p):
    p_x, p_y = p
    dof_map_internal = dof_map_crazy_ext_gauss_nodes(n, (p_x, p_y))[..., :(p_x + 1) * (p_y + 1)]
    return dof_map_internal


def dof_map_crazy_ext_gauss_edges_internal(n, p):
    """
    CrazyMesh, general numbering rule, standard extended gauss 1-form

    """
    px, py = p
    p = px + 1
    nx, ny = n
    global_numbering_internal = np.zeros(shape=(nx * ny, 2 * p * (p + 1)), dtype=np.int32)

    A = np.array([int(i) for i in range(2 * p * (p + 1))])
    for s in range(nx * ny):

        global_numbering_internal[s, 0: 2 * p * (p + 1)] = s * 2 * p * (p + 1) + A

    return global_numbering_internal


def dof_map_crazy_ext_gauss_faces_internal(n, p):
    px, py = p
    dof_map_internal = dof_map_crazy_ext_gauss_faces(n, p)
    return dof_map_internal


def dof_map_crazy_ext_gauss_faces(n, p):
    """
    CrazyMesh, general numbering rule, standard extended gauss 2-form
    """
    p = (p[0] + 2, p[1] + 2)
    global_numbering = dof_map_crazy_lobatto_faces(n, p)
    return global_numbering


if __name__ == '__main__':
    mesh = CrazyMesh(2, (2, 2), ((-1, 1), (-1, 1)))
    dof_map = dof_map_crazy_ext_gauss_edges(mesh.n, (1, 1))
