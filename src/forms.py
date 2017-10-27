import sys
import time
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from profilehooks import profile

import basis_forms
import quadrature
from basis_forms import BasisForm
from function_space import FunctionSpace
from helpers import unblockshaped
from mesh import CrazyMesh
from polynomials import edge_basis, lagrange_basis

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


class AbstractForm(ABC):
    """Abstract class for k-forms."""

    def __init__(self, *args):
        self.function_space = args[0]
        self.mesh = self.function_space.mesh
        self.p = self.function_space.p
        self._cochain = None
        self._basis = None
        if len(args) == 2:
            if args[1] is callable:
                self.function = args[1]
            elif isinstance(args[1], (np.ndarray)):
                self.cochain = args[1]
        self._cochain_local = None

    @property
    def cochain_local(self):
        """Map the cochain elements into local dof with the dof map."""
        if self._cochain_local is None:
            self._cochain_local = self.cochain[np.transpose(self.function_space.dof_map.dof_map)]
        return self._cochain_local

    @cochain_local.setter
    def cochain_local(self, cochain):
        try:
            assert np.shape(cochain)[-1] == (self.function_space.mesh.num_elements)
            self._cochain_local = cochain
        except AssertionError as e:
            raise AssertionError(
                "The number of the local cochain columns should be equal to the number of elements")

    @property
    def basis(self):
        """Return the basis function correponding to the element type."""
        if self._basis is None:
            # get name of the class of the basis functions
            elem_type = self.function_space.str_to_elem[self.function_space.form_type]
            self.basis = getattr(basis_forms, elem_type)(self.function_space)
        return self._basis

    @basis.setter
    def basis(self, basis):
        self._basis = basis

    @property
    def cochain(self):
        """Cochain relative to a form.

        It is a ndarry having for rows the dofs and in the columns the number of the element.
        """
        return self._cochain

    @cochain.setter
    def cochain(self, cochain):
        try:
            assert np.shape(cochain) == (self.function_space.num_dof,)
            self._cochain = cochain
        except AssertionError:
            raise AssertionError(
                "The dofs of the cochain do not match the dofs of the function space. \n \
                shape cochain {0}, number of degrees of freedom : {1}" .format(np.shape(cochain), self.function_space.num_dof))

    def cochain_to_global(self):
        """Map the local dofs of the cochain into the global cochain."""
        self._cochain = np.zeros((self.function_space.num_dof))
        dof_map = np.transpose(self.function_space.dof_map.dof_map)
        # reorder degrees of freedom
        for i, row in enumerate(self.cochain_local):
            for j, dof_value in enumerate(row):
                self._cochain[dof_map[i, j]] = dof_value

    def export_to_plot(self):
        """Return x, y coordinates and data to plot."""
        try:
            num_pts_y, num_pts_x = np.shape(self.basis.xi)
            num_el_x, num_el_y = self.function_space.mesh.n_x, self.function_space.mesh.n_y
            x, y = self.function_space.mesh.mapping(self.basis.xi, self.basis.eta)

            x_4d = x.reshape(num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
            x = np.moveaxis(x_4d, 2, 1).reshape(
                num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

            y_4d = y.reshape(num_pts_y, num_pts_x,  num_el_y, num_el_x, order='F')

            y = np.rollaxis(y_4d, 2, 1).reshape(
                num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

            recon_4d = self.reconstructed.reshape(
                num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
            reconstructed = np.moveaxis(recon_4d, 2, 1).ravel('F').reshape(
                num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

            return (x, y), reconstructed
        except AttributeError:
            raise AttributeError("The mesh is not crazy")

    def l_2_norm(self, func, quad='gauss'):
        """Calculate the L_2 error between a function and the reconstructed form.

        Parameters
        ----------
        func : callable
            Analytical function
        quad (optional) : quadrature type
            Quadrature used for the integration. The quadrature is fed to parse_quad_type.

        Returns
        -------
        l_2_norm : float
            L_2 norm between the reconstructed form and an analytical function
        local_error: float
            Sum of the error at the quadrature points.

        """
        self.basis.quad_grid = quad
        quad_weights_2d = np.kron(self.basis._quad_weights[0], self.basis._quad_weights[1]).reshape(
            np.size(self.basis._quad_weights[0]) * np.size(self.basis._quad_weights[1]), 1)
        # reconstruct cochain at quadrature pts
        self.reconstruct(self.basis._quad_nodes[0], self.basis._quad_nodes[1])
        pts_per_element = np.size(quad_weights_2d)
        x, y = self.function_space.mesh.mapping(self.basis.xi, self.basis.eta)
        # evaluate funcions at the domain and reshape them into shape = (num_quad_pts,num_eleßments)
        func_eval = func(x, y).reshape(
            pts_per_element, self.function_space.mesh.num_elements, order='F')
        g = self.function_space.mesh.g(self.basis.xi, self.basis.eta).reshape(
            pts_per_element, self.function_space.mesh.num_elements, order='F')
        # error at the quadrature points
        local_error = (self.reconstructed - func_eval)**2
        # integrate to get the l_2 norm
        global_error = local_error * g * quad_weights_2d
        return np.sum(global_error)**0.5, np.sum(local_error)**0.5

    @abstractmethod
    def discretize(self):
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, xi, eta):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError


class Form_0(AbstractForm):
    """Create a 0-form."""

    def __init__(self, *args):
        # TODO: after reconstruction and discretization the cochain is 2d
        super().__init__(*args)
        self.reconstructed = None

    def discretize(self, func):
        """Project a function onto a finite dimensional space of 0-forms."""
        # discretize function and reshape as (# dof per element, # num elements )
        self.cochain_local = func(*self.mesh.mapping(*self.basis.basis_nodes))
        self.cochain_to_global()

    def reconstruct(self, xi, eta):
        """Reconstruct the 0-form on the physical domain."""
        self.basis.evaluate_basis(domain=(xi, eta))
        self.reconstructed = np.tensordot(self.basis.basis, self.cochain_local, axes=([0], [0]))

    def export(self):
        raise NotImplementedError


class ExtGaussForm_0(Form_0):
    """A 0-form with extended gauss grid."""

    def __init__(self, *args):
        # TODO: after reconstruction and discretization the cochain is 2d
        super().__init__(*args)

    @property
    def cochain_local_internal(self):
        return self.cochain_local[:self.basis.num_basis]

    @property
    def cochain_internal(self):
        return self.cochain[:self.basis.num_basis * self.function_space.mesh.num_elements]

    def reconstruct(self, xi, eta):
        """Reconstruct the 0-form on the physical domain."""

        self.basis.evaluate_basis(domain=(xi, eta))
        self.reconstructed = np.tensordot(
            self.basis.basis, self.cochain_local_internal, axes=([0], [0]))

    def export(self):
        raise NotImplementedError


class Form_1(AbstractForm):
    """Class for 1-forms."""

    def __init__(self, *args):
        super().__init__(*args)
        self.reconstructed_dx = None
        self.reconstructed_dy = None

    @property
    def cochain_xi(self):
        """Return the dx component of the cochain."""
        return self._cochain[:self.basis.num_basis_xi]

    @property
    def cochain_eta(self):
        """Return the dy component of the cochain."""
        return self._cochain[-self.basis.num_basis_eta:]

    def split_cochain(self, cochain):
        """Split the cochain in the dx and dy component."""
        return cochain[:self.basis.num_basis_xi], cochain[-self.basis.num_basis_eta:]

    def discretize(self, func, quad='gauss'):
        """Discretize a vector function into a one form."""
        self.basis.quad_grid = quad
        quad, (p_x, p_y) = self.basis.quad_grid

        xi_ref, eta_ref = np.meshgrid(self.basis._quad_nodes[0], self.basis._quad_nodes[1])

        edges_size = [self.basis._edge_nodes[i][1:] - self.basis._edge_nodes[i][:-1]
                      for i in range(2)]

        magic_factor = 0.5
        cell_nodes = [(0.5 * (edges_size[i][np.newaxis, :]) *
                       (self.basis._quad_nodes[i][:, np.newaxis] + 1) + self.basis._edge_nodes[i][:-1]).ravel('F') for i in range(2)]

        quad_eta_for_dx = np.tile(self.basis._edge_nodes[1], (p_x + 1, self.p[0]))
        quad_xi_for_dx = np.repeat(cell_nodes[0].reshape(
            p_x + 1, self.p[0], order='F'), self.p[1] + 1, axis=1)

        quad_xi_for_dy = np.repeat(
            self.basis._edge_nodes[0], (p_y + 1) * self.p[1]).reshape(p_y + 1, (self.p[0] + 1) * self.p[1], order='F')
        quad_eta_for_dy = np.tile(cell_nodes[1].reshape(
            p_y + 1, self.p[1], order='F'), (1, self.p[0] + 1))

        x_dx, y_dx = self.mesh.mapping(quad_xi_for_dx, quad_eta_for_dx)

        cochain_local_xi = np.tensordot(self.basis._quad_weights[0], func[0](x_dx, y_dx) * self.mesh.dx_dxi(
            quad_xi_for_dx, quad_eta_for_dx) + func[1](x_dx, y_dx) * self.mesh.dy_dxi(quad_xi_for_dx, quad_eta_for_dx),
            axes=((0), (0))) * np.repeat(edges_size[0] * magic_factor, self.p[1] + 1).reshape(self.p[0] * (self.p[1] + 1), 1)

        x_dy, y_dy = self.mesh.mapping(quad_xi_for_dy, quad_eta_for_dy)

        cochain_local_eta = np.tensordot(self.basis._quad_weights[1], func[0](x_dy, y_dy) * self.mesh.dx_deta(
            quad_xi_for_dy, quad_eta_for_dy) + func[1](x_dy, y_dy) * self.mesh.dy_deta(
            quad_xi_for_dy, quad_eta_for_dy), axes=((0), (0))) * \
            np.tile(edges_size[1] * magic_factor, (self.p[0] + 1, 1)).reshape(self.p[1]
                                                                              * (self.p[0] + 1), 1)
        self.cochain_local = np.vstack((cochain_local_xi, cochain_local_eta))
        self.cochain_to_global()
        return quad_xi_for_dy, quad_eta_for_dy, cochain_local_eta

    def reconstruct(self, xi, eta):
        """Reconstruct the form on the computational domain.

        Given the values of the degrees of freedom in each element the function
        reconstruct the form through the basis functions.
        """
        self.basis.evaluate_basis(domain=(xi, eta))
        xi, eta = self.basis.xi.ravel('F'), self.basis.eta.ravel('F')
        cochain_xi, cochain_eta = self.split_cochain(self.cochain_local)
        g = self.mesh.g(xi, eta)

        self.reconstructed_dx = 1 / g * (
            self.mesh.dy_deta(xi, eta) * np.tensordot(self.basis.basis_xi,
                                                      cochain_xi, axes=((0), (0)))
            - self.mesh.dy_dxi(xi, eta) * np.tensordot(self.basis.basis_eta,
                                                       cochain_eta, axes=((0), (0)))
        )
        self.reconstructed_dy = 1 / g * (-self.mesh.dx_deta(xi, eta) * np.tensordot(self.basis.basis_xi, cochain_xi, axes=(
            (0), (0))) + self.mesh.dx_dxi(xi, eta) * np.tensordot(self.basis.basis_eta, cochain_eta, axes=((0), (0))))

    def export_to_plot(self):
        """Export the domain and the correspondent values for the reconstruction.

        Return
        ------
        (x, y) : tuple of ndarrays
            Contain the x and y coordinates of the domain
        reconstructed : ndarray
            the reconstrued values of the form for the domain pts
        """
        try:
            num_pts_y, num_pts_x = np.shape(self.basis.xi)
            num_el_x, num_el_y = self.function_space.mesh.n_x, self.function_space.mesh.n_y
            x, y = self.function_space.mesh.mapping(self.basis.xi, self.basis.eta)
            # print(x[0, 0, :])
            x_4d = x.reshape(num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
            x = np.moveaxis(x_4d, 2, 1).reshape(
                num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

            y_4d = y.reshape(num_pts_y, num_pts_x,  num_el_y, num_el_x, order='F')
            y = np.rollaxis(y_4d, 2, 1).reshape(
                num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

            recon_4d_dx = self.reconstructed_dx.reshape(
                num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
            reconstructed_dx = np.moveaxis(recon_4d_dx, 2, 1).ravel('F').reshape(
                num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

            recon_4d_dy = self.reconstructed_dy.reshape(
                num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
            reconstructed_dy = np.moveaxis(recon_4d_dy, 2, 1).ravel('F').reshape(
                num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')

            return (x, y), reconstructed_dx, reconstructed_dy
        except AttributeError:
            raise AttributeError("The mesh is not crazy")

    def l_2_norm(self, vector_func, quad='gauss'):
        """Calculate the L_2 error between a function and the reconstructed form.

        Parameters
        ----------
        vector_func : list or tuple of callable
            Analytical vector function having two components x nad y
        quad (optional) : quadrature type
            Quadrature used for the integration. The quadrature is fed to parse_quad_type.

        Returns
        -------
        l_2_norm : float
            L_2 norm between the reconstructed form and an analytical vector function
        local_error: float
            Sum of the error at the quadrature points.

        """
        func_x, func_y = vector_func

        self.basis.quad_grid = quad
        quad_weights_2d = np.kron(self.basis._quad_weights[0], self.basis._quad_weights[1]).reshape(
            np.size(self.basis._quad_weights[0]) * np.size(self.basis._quad_weights[1]), 1)

        # reconstruct cochain at quadrature pts
        self.reconstruct(self.basis._quad_nodes[0], self.basis._quad_nodes[1])

        pts_per_element = np.size(quad_weights_2d)
        x, y = self.function_space.mesh.mapping(self.basis.xi, self.basis.eta)
        # evaluate funcions at the domain and reshape them into shape = (num_quad_pts,num_elements)
        func_eval_x = func_x(x, y).reshape(
            pts_per_element, self.function_space.mesh.num_elements, order='F')
        func_eval_y = func_y(x, y).reshape(
            pts_per_element, self.function_space.mesh.num_elements, order='F')
        g = self.function_space.mesh.g(self.basis.xi, self.basis.eta).reshape(
            pts_per_element, self.function_space.mesh.num_elements, order='F')
        # error at the quadrature points
        local_error = (self.reconstructed_dx - func_eval_x)**2 + \
            (self.reconstructed_dy - func_eval_y)**2
        # integrate to get the l_2 norm
        #
        global_error = local_error * g * quad_weights_2d
        return np.sum(global_error)**0.5, np.sum(local_error)**0.5


class ExtGaussForm_1(Form_1):
    """1-Form with Extended gauss edges."""

    def __init___(self, *args):
        super().__init(*args)

    @property
    def cochain_xi(self):
        """Return the dx component of the cochain."""
        return self._cochain[:self.basis.num_basis_xi]

    @property
    def cochain_eta(self):
        """Return the dy component of the cochain."""
        return self._cochain[self.basis.num_basis_xi: self.basis.num_basis]

    @property
    def cochain_local(self):
        """Map the cochain elements into local dof with the dof map."""
        if self._cochain_local is None:
            self._cochain_local = self.cochain[np.transpose(
                self.function_space.dof_map.dof_map_internal)]
        return self._cochain_local[:self.basis.num_basis]

    @cochain_local.setter
    def cochain_local(self, cochain):
        # fuzzy logic now you can assing the ghost cochain as local but when you
        # access it it returns only the local. or maybe it is good
        try:
            assert np.shape(cochain)[-1] == (self.function_space.mesh.num_elements)
            self._cochain_local = cochain
        except AssertionError as e:
            raise AssertionError(
                "The number of the local cochain columns should be equal to the number of elements")

#    @property
#    def cochain_local_with_ghosts(self):
#        """Map the cochain elements into local dof with the dof map."""
#        if self._cochain_local is None:
#            self._cochain_local = self.cochain[np.transpose(self.function_space.dof_map.dof_map)]
#        return self._cochain_local
#
#    @cochain_local_with_ghosts.setter
#    def cochain_local_with_ghosts(self, cochain):
#        try:
#            assert np.shape(cochain)[-1] == (self.function_space.mesh.num_elements)
#            self._cochain_local = cochain
#        except AssertionError as e:
#            raise AssertionError(
#                "The number of the local cochain columns should be equal to the number of elements")
#

    def cochain_to_global(self):
        """Map the local dofs of the cochain into the global cochain."""
        self.cochain = np.zeros((self.function_space.num_dof))
        if np.shape(self.cochain_local) == (self.basis.num_basis, self.function_space.mesh.num_elements):
            dof_map = np.transpose(self.function_space.dof_map.dof_map[:, :self.basis.num_basis])
        else:
            dof_map = np.transpose(self.function_space.dof_map.dof_map)
        # reorder degrees of freedom
        for i, row in enumerate(self.cochain_local):
            for j, dof_value in enumerate(row):
                self.cochain[dof_map[i, j]] = dof_value

    def split_cochain(self, cochain):
        """Split the cochain in the dx and dy component."""
        return cochain[:self.basis.num_basis_xi], cochain[self.basis.num_basis_xi: self.basis.num_basis]


class Form_2(AbstractForm):
    """Class of two forms."""

    def __init__(self, *args):
        super().__init__(*args)
        self.reconstructed = None

    def discretize(self, func, quad='gauss'):
        # TODO: change self p by len(self._face_nodes)
        """Project a function into a finite element space of 2-forms.

        The projection is done in the reference element. It follows the inverse of the pullback to project into the physical domain.
        """
        # calculate quadrature nodes and weights
        self.basis.quad_grid = quad
        quad, (p, _) = self.basis.quad_grid
        xi_ref, eta_ref = np.meshgrid(self.basis._quad_nodes[0], self.basis._quad_nodes[1])
        quad_weights = np.kron(self.basis._quad_weights[0], self.basis._quad_weights[1])

        # calculate the dimension of the edges of the cells
        dim_faces = [self.basis._face_nodes[i][1:] - self.basis._face_nodes[i][:-1]
                     for i in range(2)]
        # set up the right amout of x and y dimensions of the edges of the cell
        x_dim = np.repeat(dim_faces[0], self.p[1])
        y_dim = np.tile(dim_faces[1], self.p[0])
        magic_factor = 0.25
        cell_nodes = [(0.5 * (dim_faces[i][np.newaxis, :]) *
                       (self.basis._quad_nodes[i][:, np.newaxis] + 1) + self.basis._face_nodes[i][:-1]).ravel('F') for i in range(2)]

        cell_area = np.diag(magic_factor * x_dim * y_dim)
        # xi coordinates of the quadrature nodes
        # in the column are stored the coordinates of the quad points for contant xi for all faces
        xi = np.repeat(np.repeat(cell_nodes[0], (p + 1)
                                 ).reshape((p + 1)**2, self.p[0], order='F'), self.p[1], axis=1)

        eta = np.tile(np.tile(cell_nodes[1].reshape(
            p + 1, self.p[1], order='F'), (p + 1, 1)), self.p[0])

        # map onto the physical domain and compute the Jacobian
        x, y = self.mesh.mapping(xi, eta)
        g = self.mesh.g(xi, eta)

        # compute the cochain integrating and then applying inverse pullback
        self.cochain_local = np.sum(np.tensordot(quad_weights, (func(x, y) * g),
                                                 axes=((0), (0))) * cell_area[:, :, np.newaxis], axis=0)
        self.cochain_to_global()

    def reconstruct(self, xi, eta):
        """Reconstruct a cochain into a 2-Form.

        The reconstructed form is stored into the attribute reconstructed

        Parameters
        ----------
        xi : array
            1D array with the xi coordinates of the domain of reconstruction on the ref. element
        eta : array
            1D array with the eta coordinates of the domain of reconstruction on the ref. element

        """
        self.basis.evaluate_basis(domain=(xi, eta))
        xi, eta = self.basis.xi.ravel('F'), self.basis.eta.ravel('F')
        self.reconstructed = np.tensordot(
            self.basis.basis, self.cochain_local, axes=((0), (0))) / self.mesh.g(xi, eta)


class ExtGaussForm_2(Form_2):
    def __init___(self, *args):
        super().__init(*args)

    def discretize(self, func, quad='gauss'):
        """Project a function into a finite element space of 2-forms.

        The projection is done in the reference element. It follows the inverse of the pullback to project into the physical domain.
        """
        # calculate quadrature nodes and weights
        self_p = (self.p[0] + 2, self.p[1] + 2)
        self.basis.quad_grid = quad
        quad, (p, _) = self.basis.quad_grid
        xi_ref, eta_ref = np.meshgrid(self.basis._quad_nodes[0], self.basis._quad_nodes[1])
        quad_weights = np.kron(self.basis._quad_weights[0], self.basis._quad_weights[1])

        # calculate the dimension of the edges of the cells
        dim_faces = [self.basis._face_nodes[i][1:] - self.basis._face_nodes[i][:-1]
                     for i in range(2)]

        # set up the right amout of x and y dimensions of the edges of the cell
        x_dim = np.repeat(dim_faces[0], self_p[1])

        y_dim = np.tile(dim_faces[1], self_p[0])
        magic_factor = 0.25
        cell_nodes = [(0.5 * (dim_faces[i][np.newaxis, :]) *
                       (self.basis._quad_nodes[i][:, np.newaxis] + 1) + self.basis._face_nodes[i][:-1]).ravel('F') for i in range(2)]

        cell_area = np.diag(magic_factor * x_dim * y_dim)

        # xi coordinates of the quadrature nodes
        # in the column are stored the coordinates of the quad points for contant xi for all faces
        xi = np.repeat(np.repeat(cell_nodes[0], (p + 1)
                                 ).reshape((p + 1)**2, self_p[0], order='F'), self_p[1], axis=1)

        eta = np.tile(np.tile(cell_nodes[1].reshape(
            p + 1, self_p[1], order='F'), (p + 1, 1)), self_p[0])

        # map onto the physical domain and compute the Jacobian
        x, y = self.mesh.mapping(xi, eta)
        g = self.mesh.g(xi, eta)

        # compute the cochain integrating and then applying inverse pullback
        self.cochain_local = np.sum(np.tensordot(quad_weights, (func(x, y) * g),
                                                 axes=((0), (0))) * cell_area[:, :, np.newaxis], axis=0)
        self.cochain_to_global()


def Form(*args):
    """Wrap around the classes of forms."""
    # TODO: create disctionary for forms
    return getattr(sys.modules[__name__], args[0].str_to_form[args[0].form_type])(*args)


def cochain_to_global(function_space, cochain_local):
    """Map the local dofs of the cochain into the global cochain."""
    cochain = np.zeros(((function_space.num_dof)))
    dof_map = np.transpose(function_space.dof_map.dof_map)
    # reorder degrees of freedom
    for i, row in enumerate(cochain_local):
        try:
            for j, dof_value in enumerate(row):
                cochain[dof_map[i, j]] = dof_value
        except TypeError as t:
            raise TypeError("The cochain provided in cochain_to_global is one dimensional")
    return cochain


def cochain_to_local(function_space, cochain):
    """Map the cochain elements into local dof with the dof map."""
    return cochain[np.transpose(function_space.dof_map.dof_map)]


def func(x, y):
    return (x + y) / (x + y)


if __name__ == "__main__":
#    p_s = [(2, 2)]
#    n = (1, 1)
#    for p in p_s:
#        mesh = CrazyMesh(2, n, ((-1, 1), (-1, 1)), 0.0)
#        func_space = FunctionSpace(mesh, '1-lobatto', p)
#        form_1 = Form(func_space)
#        form_1.discretize((func, func))

    # p = 3, 3
    # nx, ny = 2, 2
    crazy_mesh = CrazyMesh(2, (1, 1), ((-1, 1), (-1, 1)), curvature=0)
    function_space = FunctionSpace(crazy_mesh, '0-gauss', 0)
    f0 = Form(function_space)
    def p(x,y): return np.sin(np.pi*x) * np.cos(np.pi*y)
    f0.discretize(p)
    print(f0.basis.inner(f0.basis))
    
    # cochain = np.ones(function_space.num_dof)
    # print(cochain)
    # # basis = BasisForm(function_space)
    # form_1 = Form(function_space, cochain)
    # xi = eta = np.linspace(-1, 1, 5)
    # xi = eta = quadrature.lobatto_quad(3)[0]
    # form_1.reconstruct(xi, eta)
    # print(form_1.reconstructed)
    # print(np.shape(form_1.reconstructed))
    #
    # form_1.reconstructed_dx = np.arange(50).reshape(25, 2)
    # print(form_1.reconstructed_dx)
#    form_1.plot()
    # np.__config__.show()
    # function_space = FunctionSpace(crazy_mesh, '0-lobatto', p)
    # form_2 = Form(function_space)
    # form_2.discretize(func)
    # print(form_2.cochain)
