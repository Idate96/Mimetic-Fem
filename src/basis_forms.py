"""This module contain classes to ebastract basis functions.

There is an abstract class to containing all the basic properties and methods.
Then for each kind of form exists a dedicated class for the correponding
basis functions.

"""
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from exceptions import MeshError, QuadratureError
from multiprocessing import Process
from inner_product import MeshFunction
import matplotlib.pyplot as plt
import numpy as np
# from profilehooks import profile
import scipy.sparse as sparse

import quadrature
from function_space import FunctionSpace
from helpers import beartype, runInParallel
from inner_product import *
from mesh import AbstractMesh, CrazyMesh
from polynomials import edge_basis, lagrange_basis


# os.system("taskset -p 0xff %d" % os.getpid())


class AbstractBasisForm(ABC):
    """Abstract class for basis forms.

    Parameters
    ----------
    function_space : FunctionSpace
        Function space associated with the basis form.

    Attributes
    ----------
    function_space : FunctionSpace
        Function space associated with the basis form.

    Methods
    -------
    interpolate
    reconstruct
    save
    export
    _inner_consistency_check
        check the compatibility of two basis forms to perform the inner product

    """

    def __init__(self, function_space):
        self.function_space = function_space
        self.mesh = function_space.mesh
        # TODO: init here basis and quad

    @property
    def mesh(self):
        """AbstractMesh : mesh."""
        return self.function_space.mesh

    @mesh.setter
    @beartype
    def mesh(self, mesh: AbstractMesh):
        self._mesh = mesh

    @property
    def basis(self):
        """ndarray: values of the basis functions at the degrees of freedom.

        The ith rows contain all the value of the ith basis functions
        at all the degree of freedom
        """
        # lazy evaluation
        if self._basis is None:
            self.evaluate_basis()
        return self._basis

    @basis.setter
    def basis(self, basis):
        self._basis = basis

    @property
    def quad_grid(self):
        """Quad grid specifies the qudrature type.

        Available: lobatto, gauss, extended_gauss.
        """
        try:
            p_int = tuple([np.size(self._quad_nodes[i]) - 1 for i in range(2)])
        except TypeError as e:
            raise TypeError("Quad grid not yet defined")
        return self._quad_grid, p_int

    @quad_grid.setter
    def quad_grid(self, quad_type):
        # parse user inputs
        quad_grid, p = parse_grid_input(quad_type, self.p)
        self._quad_grid = quad_grid
        # init lists of nodes and weights
        self._quad_nodes = [0, 0]
        self._quad_weights = [0, 0]
        # create the nodes and the weights using quadrature methods
        self._quad_nodes[0], self._quad_weights[0] = getattr(
            quadrature, quad_grid[0] + '_quad')(p[0])
        self._quad_nodes[1], self._quad_weights[1] = getattr(
            quadrature, quad_grid[1] + '_quad')(p[1])

    @abstractmethod
    def evaluate_basis(self, xi, eta):
        """Abstract method used to interpolate the basis funtions
         at the reference element.
        """
        raise NotImplementedError

    def save(self):
        """Abstract method to save the current case in a pickle file."""
        raise NotImplementedError

    def export(self):
        """Abstract method to export basis function in csv file."""
        raise NotImplementedError

    def _inner_consistency_check(self, other):
        """Check consistency for inner product.

        In order to perform the inner product the basis functions
        have to be evaluated at the same point, furthermore a unique quadrature rule is used.
        Finally, they need to share the same mesh.

        Parameters
        ----------
        other : BasisForm
            Basis form used to perform the inner product

        """
        # check quadrature rules
        if self.quad_grid is None or other.quad_grid is None:
            raise QuadratureError("Test function quadrature: {0}, and the trial one: {1}, are not valid" .format(
                self.quad_grid, other.quad_grid))
        # check location dof of the forms
        if self.quad_grid != other.quad_grid:
            raise QuadratureError("The quadrature for the test and trial function " +
                                  "do not match: {0} != {1} " .format(self.quad_grid, other.quad_grid))
        # chack mesh
        if self.mesh is not other.mesh:
            raise MeshError("Mesh of the forms do not match")


def parse_grid_input(grid_input, standard_p):
    """Parse the user input for the generation of grids both for dofs and quadrature."""
    if isinstance(grid_input, (str)):
        # same grid for x and y
        grid = (grid_input, grid_input)
        # if degree not given use the standard one
        p = standard_p
    # if also the degree is specified
    elif isinstance(grid_input, (tuple)):
        # if same grid in x and y
        if isinstance(grid_input[0], (str)):
            grid = grid_input[0], grid_input[0]
        # if diffent grid in x and y
        elif isinstance(grid_input[0], (tuple)):
            grid = grid_input[0][0], grid_input[0][1]
        # if same p for x and y
        if isinstance(grid_input[1], (int)):
            p = grid_input[1], grid_input[1]
        # if different p for x and y
        elif isinstance(grid_input[1], (tuple)):
            p = grid_input[1][0], grid_input[1][1]
    try:
        return grid, p
    except UnboundLocalError as e:
        raise UnboundLocalError("Quadrature grid input is not valid")


class BasisForm_0(AbstractBasisForm):
    """Defines an element with 0-form basis functions.

    The element contains the mesh, the dofs and a set of basis functions.
    Currently only quadrilateral are supported.

    Parameters
    ----------
    *args
        function space must be args[0]. The other arguments are optional.
        the type of nodal grid is args[1], available : lobatto, gauss, extended_gauss
        the type of quadrature is args[2], available : lobatto, gauss, extended_gauss

    Attributes
    ----------
    p : tuple
        Value of the polinomial degree for x and y direction
    nodal_nodes: array
        Coordinates of the points of the dofs
    num_basis_funcs: int
        Number of the basis functions (= num_dof)
    quad_grid : str, list
        Kind of the quadrature
    quad_nodes : array
        Array of the quadrature nodes
    quad_weights : array
        Array of the quadrature weights
    xi : ndarray
        Coordinates of the x position of the domain in the reference element at which the basis fucntions are evaluated
    eta : ndarray
        Coordinates of the y position of the domain in the reference element at which the basis fucntions are evaluated
    basis : ndarray
        2d array containing on the rows different basis functions and on the columns
        their values at the dof point, quadrature points or domain if stated.

    Methods
    -------
    interpolate
    reconstruct
    export
    save
    inner

    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            function_space = args[0]

        super().__init__(function_space)

        self.p = function_space.p

        self.num_basis = (self.p[0] + 1) * (self.p[1] + 1)

        self._nodal_nodes = None
        self._nodal_grid = None

        self._quad_nodes = None
        self._quad_weights = None
        self._quad_grid = None

        self.xi, self.eta = None, None
        self._basis = None
        self._basis_nodes = None

    @property
    def nodal_grid(self):
        """Nodal grid specifies the kind of the nodal dof.

        Available : lobatto, gauss, extended_gauss.
        """
        return self._nodal_grid

    @nodal_grid.setter
    def nodal_grid(self, grid):
        self._nodal_grid = grid

    @property
    def basis_nodes(self):
        """2D coordinates of degrees of freedom.

        Basis nodes is a tuple containing two 1D array. The two array are the F
        flattened cooridinates of the dofs."""
        if self._basis_nodes is None:
            xi, eta = np.meshgrid(self._nodal_nodes[0], self._nodal_nodes[1])
            self._basis_nodes = (xi.ravel('F'), eta.ravel('F'))
        return self._basis_nodes

    @basis_nodes.setter
    def basis_nodes(self, basis_nodes):
        self._basis_nodes = basis_nodes

        # self.xi, self.eta = np.meshgrid(self._quad_nodes[0], self._quad_nodes[1])

    def __getitem__(self, indexes):
        """Give value of the basis function.

        Returns the value of ith basis function at the jth point
        of the prescribed domain ie basis[i,j].
        """
        i, j = indexes
        try:
            return self.basis[i, j]
        except IndexError as e:
            raise IndexError("Index {0} out of range, the number of basis functions is {1}."
                             .format(i, self.num_basis))

    def __call__(self, xi, eta):
        """Return the evaluation of the basis."""
        domain = (xi, eta)
        nodal_basis_1d = [lagrange_basis(
            self._nodal_nodes[i], domain[i]) for i in range(1)]
        basis = np.kron(nodal_basis_1d[0], nodal_basis_1d[1])
        return basis

    def evaluate_basis(self, domain=None):
        """Evaluate the basis.

        The basis are evaluated in at the position of the dof or quad nodes (if supplied) or
        at the domain specified.
        """
        if domain is None:
            # evaluate the basis functions at dof or quad points
            if self._quad_nodes is None:
                # evaluate basis at dof points
                warnings.warn("Quadrature for {0} not specified, basis functions evaluated at grid points" .format(
                    self.__class__), UserWarning, stacklevel=2)
                # evaluate the lagrange basis in one 1d for both x and y at nodal points
                nodal_basis_1d = [lagrange_basis(
                    self._nodal_nodes[i], self._nodal_nodes[i]) for i in range(2)]
                # store domain
                self.xi, self.eta = np.meshgrid(self._nodal_nodes[0], self._nodal_nodes[1])
            else:
                # evaluate the lagrange basis in one 1d for both x and y at quad points
                nodal_basis_1d = [lagrange_basis(
                    self._nodal_nodes[i], self._quad_nodes[i]) for i in range(2)]
                # stor domain
                self.xi, self.eta = np.meshgrid(self._quad_nodes[0], self._quad_nodes[1])
            # tensor product
        else:
            self.xi, self.eta = np.meshgrid(*domain)
            nodal_basis_1d = [lagrange_basis(
                self._nodal_nodes[i], domain[i]) for i in range(2)]
        self.basis = np.kron(nodal_basis_1d[0], nodal_basis_1d[1])

    # @profile
    def inner(self, other):
        """Compute inner product.

        Computes the inner product on the reference domain between two basis functions of the
        same form degree. The evaluation points of the basis functions should be same.
        The quadrature rule used, is the one belonging to the test basis (self).

        The inner product on the physical domain is computed in the reference domain by the application of the pullback on the forms.

        Parameters
        ----------
        other : BasisForm_0
            A 0-form basis function

        Returns
        -------
        M_0 : ndarray
            3-d inner product matrix. M_0[i,j,k] returns the inner product value between the i th and j th basis functions of the k th element.

        """
        # check consistency of discretization
        self._inner_consistency_check(other)
        # generate 2d quadrature points and weights
        self.xi, self.eta = np.meshgrid(self._quad_nodes[0], self._quad_nodes[1])
        quad_weights_2d = np.kron(self._quad_weights[0], self._quad_weights[1]).reshape(
            np.size(self._quad_weights[0]) * np.size(self._quad_weights[1]), 1)
        # det of the jacobian
        g = self.mesh.g(self.xi.ravel('F'), self.eta.ravel('F'))
        # inner product
        M_0 = np.dot(self.basis, other.basis[:, :, np.newaxis]
                     * (quad_weights_2d * g)[np.newaxis, :, :])
        return M_0

    def wedged(self, other):
        """Integrate the wedge product of two basis.

        The 0 forms is the trial function (on the columns) and the 2-form is the test function.

        """
        assert self.function_space.k + \
            other.function_space.k == self.mesh.dim, 'k-form wedge l-form, k+l should be equal to n'

        p_0 = np.max([self.p[0], other.p[0]])
        p_1 = np.max([self.p[1], other.p[1]])
        self.quad_grid = ('gauss', 'gauss'), (p_0, p_1)
        other.quad_grid = ('gauss', 'gauss'), (p_0, p_1)

        quad_weights_2d = np.kron(self._quad_weights[0], self._quad_weights[1]
                                  ).reshape(1,
                                            np.size(self._quad_weights[0]) * np.size(self._quad_weights[1]))

        # W = np.zeros((other.num_basis, self.num_basis))
        W = np.tensordot(other.basis, self.basis * quad_weights_2d, axes=((1), (1)))
        return W


class LobattoNodal(BasisForm_0):
    """Element with 0-form basis functions with Gauss lobatto points as dof."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodal_grid = 'lobatto'
        # self._nodal_nodes = [getattr(quadrature, self.nodal_grid + '_quad')(p)[0] for p in self.p]

    @BasisForm_0.nodal_grid.setter
    def nodal_grid(self, nodal_grid):
        self._nodal_grid = nodal_grid
        self._nodal_nodes = [getattr(quadrature, self._nodal_grid + '_quad')(p)[0] for p in self.p]


class GaussNodal(BasisForm_0):
    """Element with 0-form basis functions with Gauss points as dof."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodal_grid = 'gauss'
        # self._nodal_nodes = [getattr(quadrature, self.nodal_grid + '_quad')(p)[0] for p in self.p]

    @BasisForm_0.nodal_grid.setter
    def nodal_grid(self, nodal_grid):
        self._nodal_grid = nodal_grid
        self._nodal_nodes = [getattr(quadrature, self._nodal_grid + '_quad')(p)[0] for p in self.p]


class ExtGaussNodal(BasisForm_0):
    """Element with 0-form basis functions with Gauss points as dof."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodal_grid = 'gauss'
        # self._nodal_nodes = [getattr(quadrature, self.nodal_grid + '_quad')(p)[0] for p in self.p]

    @BasisForm_0.nodal_grid.setter
    def nodal_grid(self, nodal_grid):
        self._nodal_grid = nodal_grid
        self._nodal_nodes = [getattr(quadrature, self._nodal_grid + '_quad')(p)[0] for p in self.p]

    @property
    def basis_nodes(self):
        if self._basis_nodes is None:
            xi, eta = np.meshgrid(self._nodal_nodes[0], self._nodal_nodes[1])
            xi, eta = xi.ravel('F').tolist(), eta.ravel('F').tolist()

            # nodes on left boundary
            xi.extend([-1] * (self.p[1] + 1))
            eta.extend(self._nodal_nodes[1])

            # nodes on right boundary
            xi.extend([+1] * (self.p[1] + 1))
            eta.extend(self._nodal_nodes[1])

            # nodes on bottom boundary
            xi.extend(self._nodal_nodes[0])
            eta.extend([-1] * (self.p[0] + 1))

            # nodes on top boundary
            xi.extend(self._nodal_nodes[0])
            eta.extend([+1] * (self.p[0] + 1))

            self.basis_nodes = (np.asarray(xi), np.asarray(eta))

        return self._basis_nodes

    @basis_nodes.setter
    def basis_nodes(self, basis_nodes):
        self._basis_nodes = basis_nodes


class FlexibleNodal(BasisForm_0):
    """Element with 0-form basis function with flexible dof"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodal_grid = self.function_space.grid_type

    @BasisForm_0.nodal_grid.setter
    def nodal_grid(self, nodal_grid):
        self._nodal_grid = nodal_grid
        self._nodal_nodes = [getattr(quadrature, nodal_grid[i] + '_quad')
                             (self.p[i])[0] for i in range(2)]


class BasisForm_1(AbstractBasisForm):
    """Defines an element with 1-form basis functions."""

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            function_space = args[0]
        super().__init__(function_space)
        self.p = function_space.p

        self._nodal_grid = None
        self._nodal_nodes = None

        self._edge_grid = None
        self._edge_nodes = None

        # quad nodes are defined only for x and y direction
        self._quad_grid = None
        self._quad_nodes = None
        self._quad_weights = None

        self.xi, self.eta = None, None
        self._basis = None

        self.num_basis_xi = self.p[0] * (self.p[1] + 1)
        self.num_basis_eta = self.p[1] * (self.p[0] + 1)
        self.num_basis = self.num_basis_xi + self.num_basis_eta

        self.quad_grid = ('gauss', 'gauss'), (self.p[0], self.p[1])

    @property
    def basis(self):
        # TODO: possible refactoring to shift these propeties onto basisform
        """ndarray: values of the basis functions at the degrees of freedom.

        The ith rows contain all the value of the ith basis functions
        at all the degree of freedom
        """
        if self._basis is None:
            self.evaluate_basis()
        return self._basis

    @basis.setter
    def basis(self, basis):
        self._basis = basis

    @property
    def basis_xi(self):
        """Return the basis related to the dx component of the 1-form."""
        return self.basis[:self.num_basis_xi]

    @property
    def basis_eta(self):
        """Return the basis related to the dy component of the 1-form."""
        return self.basis[-self.num_basis_eta:]

    @property
    def nodal_grid(self):
        """Nodal grid specifies the kind of the nodal dof.

        Available : lobatto, gauss, extended_gauss.
        """
        return self._nodal_grid

    @nodal_grid.setter
    def nodal_grid(self, grid):
        self._nodal_grid = grid

    @property
    def edge_grid(self):
        """Edge grid specifies the kind of the edge dof.

        Available : lobatto, gauss, extended_gauss.
        """
        return self._edge_grid

    @edge_grid.setter
    def edge_grid(self, grid):
        self._edge_grid = grid

    def __call__(self, xi, eta):
        # TODO do not call interpolate otherwise you overwrite the attributes
        self.interpolate(domain=(xi, eta))
        return self.edge_basis

    def __getitem__(self, tup):
        i, j = tup
        if i < self.num_basis_xi:
            return self.edge_basis[0][i, j]
        else:
            if i > self.num_basis:
                raise IndexError("Index {0} out of range, the number of basis functions is {1}."
                                 .format(i, self.num_basis))
            return self.edge_basis[1][i % self.num_basis_xi, j]

    def evaluate_basis(self, domain=None):
        """Evaluate the basis.

        The basis are evaluated in at the position of the dof or quad nodes (if supplied) or
        at the domain specified.
        """
        if domain is None:
            # evaluate the basis functions at dof or quad points
            if self._quad_nodes is None:
                # evaluate basis at dof points
                warnings.warn("Quadrature for {0} not specified, basis functions evaluated at grid points" .format(
                    self.__class__), UserWarning, stacklevel=2)

                nodal_basis_1d = [lagrange_basis(
                    self._nodal_nodes[i], self._nodal_nodes[i]) for i in range(2)]
                edge_basis_1d = [edge_basis(self._edge_nodes[i], self._edge_nodes[i])
                                 for i in range(2)]

                self.basis = np.zeros(
                    (self.num_basis, np.size(self._nodal_nodes[0]) * np.size(self._edge_nodes[0])))
            else:
                # nodal basis : first entry of nodal basis for y second for x
                # quad: first entry for x  and second for y
                # edge basis : first entry for x and second for y
                nodal_basis_1d = [lagrange_basis(
                    self._nodal_nodes[i], self._quad_nodes[1 - i]) for i in range(2)]
                edge_basis_1d = [edge_basis(self._edge_nodes[i], self._quad_nodes[i])
                                 for i in range(2)]
                self.basis = np.zeros((self.num_basis, np.size(
                    self._quad_nodes[0]) * np.size(self._quad_nodes[1])))

        else:
            self.xi, self.eta = np.meshgrid(domain[0], domain[1])
            nodal_basis_1d = [lagrange_basis(
                self._nodal_nodes[i], domain[1 - i]) for i in range(2)]
            edge_basis_1d = [edge_basis(self._edge_nodes[i], domain[i])
                             for i in range(2)]
            self.basis = np.zeros((self.num_basis, np.size(domain[0]) * np.size(domain[1])))
        # allocate the values
        self.basis[:self.num_basis_xi] = np.kron(edge_basis_1d[0], nodal_basis_1d[0])
        self.basis[-self.num_basis_eta:] = np.kron(nodal_basis_1d[1], edge_basis_1d[1])

    def weighted_metric_tensor(self, xi, eta, K):
        """Calculate the metric tensor weighted with the constitutive law."""
        K.eval_tensor(xi, eta)

        if self.function_space.is_inner:
            k_11, k_12, k_22 = K.tensor
        else:
            k_11, k_12, k_22 = K.inverse

        dx_deta = self.mesh.dx_deta(xi, eta)
        dx_dxi = self.mesh.dx_dxi(xi, eta)
        dy_deta = self.mesh.dy_deta(xi, eta)
        dy_dxi = self.mesh.dy_dxi(xi, eta)
        g = dx_dxi * dy_deta - dx_deta * dy_dxi
        g_11 = (dx_deta**2 * k_11 +
                2 * dy_deta * dx_deta * k_12 +
                dy_deta**2 * k_22) / g

        g_12 = (dx_dxi * dx_deta * k_11 +
                (dy_dxi * dx_deta + dx_dxi
                 * dy_deta) * k_12 + dy_dxi * dy_deta * k_22) / g

        g_22 = (dx_dxi**2 *
                k_11 + 2 * dy_dxi * dx_dxi * k_12 + dy_dxi**2 * k_22) / g
        return g_11, g_12, g_22

    def inner(self, other, K=None):
        """Calculate inner product between 1-forms."""
        # TODO: write analitical experession in docstring
        # check consistency of the discretization
        self._inner_consistency_check(other)
        self.xi, self.eta = np.meshgrid(self._quad_nodes[0], self._quad_nodes[1])
        quad_weights_2d = np.kron(self._quad_weights[0], self._quad_weights[1]).reshape(
            np.size(self._quad_weights[0]) * np.size(self._quad_weights[1]), 1)

        # calculate metric componets
        if K is not None:
            # metric tensor weighted by constitutive laws
            g_11, g_12, g_22 = self.weighted_metric_tensor(
                self.xi.ravel('F'), self.eta.ravel('F'), K)
        else:
            # usual metric terms
            g_11, g_12, g_22 = self.mesh.metric_tensor(self.xi.ravel('F'), self.eta.ravel('F'))

        M_1 = np.zeros((self.num_basis, other.num_basis,
                        self.mesh.num_elements))
        M_1[:self.num_basis_xi, :other.num_basis_xi, :] = np.tensordot(
            self.basis_xi, other.basis_xi[:, :, np.newaxis] * (g_11 *
                                                               quad_weights_2d), axes=((1), (1)))
        M_1[:self.num_basis_xi, -other.num_basis_eta:, :] = np.tensordot(
            self.basis_xi, other.basis_eta[:, :, np.newaxis] * (g_12 * quad_weights_2d), axes=((1), (1)))
        M_1[-self.num_basis_eta:, :other.num_basis_xi] = np.transpose(
            M_1[:self.num_basis_xi, -other.num_basis_eta:, :], (1, 0, 2))
        M_1[-self.num_basis_eta:, -other.num_basis_eta:] = np.tensordot(
            self.basis_eta, other.basis_eta[:, :, np.newaxis] * (g_22 *
                                                                 quad_weights_2d), axes=((1), (1)))
        return M_1

    def wedged(self, other):
        """Integrate the wedge product of two basis."""
        assert self.function_space.k + \
            other.function_space.k == self.mesh.dim, 'k-form wedge l-form, k+l should be equal to n'

        p_0 = np.max([self.p[0], other.p[0]])
        p_1 = np.max([self.p[1], other.p[1]])
        self.quad_grid = ('gauss', 'gauss'), (p_0, p_1)
        other.quad_grid = ('gauss', 'gauss'), (p_0, p_1)

        quad_weights_2d = np.kron(self._quad_weights[0], self._quad_weights[1]
                                  ).reshape(1,
                                            np.size(self._quad_weights[0]) * np.size(self._quad_weights[1]))

        W = np.zeros((other.num_basis, self.num_basis))

        W[:other.num_basis_xi, -self.num_basis_eta:] =  \
            np.tensordot(other.basis_xi, self.basis_eta * quad_weights_2d, axes=((1), (1)))
        W[-other.num_basis_eta:, :self.num_basis_xi] =  \
            -np.tensordot(other.basis_eta, self.basis_xi * quad_weights_2d, axes=((1), (1)))
        return W

    def interior_product(self, other):
        # selecting uniform evaluation points
        p_0 = np.max([self.p[0], other.p[0]])
        p_1 = np.max([self.p[1], other.p[1]])
        self.quad_grid = ('gauss', 'gauss'), (p_0, p_1)
        other.quad_grid = ('gauss', 'gauss'), (p_0, p_1)


class LobattoEdge(BasisForm_1):
    """Basis 1-form having as dof the values at edges resulting from a Lobatto discretization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodal_grid = 'lobatto', 'lobatto'
        self.edge_grid = 'lobatto', 'lobatto'

    # TODO: inherit the standard method
    @BasisForm_1.nodal_grid.setter
    def nodal_grid(self, nodal_grid):
        self._nodal_grid = nodal_grid
        self._nodal_nodes = [getattr(quadrature, self._nodal_grid[i] +
                                     '_quad')(self.p[1 - i])[0] for i in range(2)]

    @BasisForm_1.edge_grid.setter
    def edge_grid(self, edge_grid):
        self._edge_grid = edge_grid
        self._edge_nodes = [getattr(quadrature, self._edge_grid[i] +
                                    '_quad')(self.p[i])[0] for i in range(2)]


class GaussEdge(BasisForm_1):
    """Basis 1-form having as dof the values at edges resulting from a Gauss discretization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodal_grid = 'gauss', 'gauss'
        self.edge_grid = 'gauss', 'gauss'

    @BasisForm_1.nodal_grid.setter
    def nodal_grid(self, nodal_grid):
        self._nodal_grid = nodal_grid
        self._nodal_nodes = [getattr(quadrature, self._nodal_grid[i] +
                                     '_quad')(self.p[1 - i])[0] for i in range(2)]

    @BasisForm_1.edge_grid.setter
    def edge_grid(self, edge_grid):
        self._edge_grid = edge_grid
        self._edge_nodes = [getattr(quadrature, self._edge_grid[i] +
                                    '_quad')(self.p[i])[0] for i in range(2)]


class ExtGaussEdge(BasisForm_1):
    """Basis 1-form having as dof the values at edges resulting from a Gauss discretization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodal_grid = 'gauss', 'gauss'
        self.edge_grid = 'extended_gauss', 'extended_gauss'

        self.num_basis = (self.p[0] + 2) * (self.p[1] + 1) + (self.p[1] + 2) * (self.p[0] + 1)
        self.num_basis_xi = (self.p[0] + 2) * (self.p[1] + 1)
        self.num_basis_eta = (self.p[1] + 2) * (self.p[0] + 1)

    @BasisForm_1.nodal_grid.setter
    def nodal_grid(self, nodal_grid):
        self._nodal_grid = nodal_grid
        self._nodal_nodes = [getattr(quadrature, self._nodal_grid[i] +
                                     '_quad')(self.p[1 - i])[0] for i in range(2)]

    @BasisForm_1.edge_grid.setter
    def edge_grid(self, edge_grid):
        self._edge_grid = edge_grid
        self._edge_nodes = [getattr(quadrature, self._edge_grid[i] +
                                    '_quad')(self.p[i])[0] for i in range(2)]


class TotalExtGaussEdge(BasisForm_1):
    """Basis 1-form having as dof the values at edges resulting from a Gauss discretization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodal_grid = 'extended_gauss', 'extended_gauss'
        self.edge_grid = 'extended_gauss', 'extended_gauss'

        self.num_basis = (self.p[0] + 2) * (self.p[1] + 3) + (self.p[1] + 2) * (self.p[0] + 3)
        self.num_basis_xi = (self.p[0] + 2) * (self.p[1] + 3)
        self.num_basis_eta = (self.p[1] + 2) * (self.p[0] + 3)

    @BasisForm_1.nodal_grid.setter
    def nodal_grid(self, nodal_grid):
        self._nodal_grid = nodal_grid
        self._nodal_nodes = [getattr(quadrature, self._nodal_grid[i] +
                                     '_quad')(self.p[1 - i])[0] for i in range(2)]

    @BasisForm_1.edge_grid.setter
    def edge_grid(self, edge_grid):
        self._edge_grid = edge_grid
        self._edge_nodes = [getattr(quadrature, self._edge_grid[i] +
                                    '_quad')(self.p[i])[0] for i in range(2)]


class BasisForm_2(AbstractBasisForm):
    """Define an element with 2-form basis functions."""

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            function_space = args[0]
        super().__init__(function_space)
        self.p = function_space.p

        self.num_basis = self.p[0] * self.p[1]

        self._face_nodes = None
        self._face_grid = None

        self.xi, self.eta = None, None

        self._basis = None

    @property
    def basis(self):
        """ndarray: values of the basis functions at the degrees of freedom.

        The ith rows contain all the value of the ith basis functions
        at all the degree of freedom
        """
        # lazy evaluation
        if self._basis is None:
            self.evaluate_basis()
        return self._basis

    @basis.setter
    def basis(self, basis):
        self._basis = basis

    @property
    def face_grid(self):
        """Nodal grid specifies the kind of the nodal dof.

        Available : lobatto, gauss, extended_gauss.
        """
        return self._face_grid

    @face_grid.setter
    def face_grid(self, grid):
        self._face_grid = grid

    def __getitem__(self, indexes):
        i, j = indexes
        try:
            return self.basis[i, j]
        except IndexError as e:
            raise IndexError("Index {0} out of range, the number of basis functions is {1}."
                             .format(i, self.num_basis))

    def evaluate_basis(self, domain=None):
        """Evaluate the basis.

        The basis are evaluated in at the position of the dof or quad nodes (if supplied) or
        at the domain specified.
        """
        if domain is None:
            # evaluate the basis functions at dof or quad points
            if self._quad_nodes is None:
                # evaluate basis at dof points
                warnings.warn("Quadrature for {0} not specified, basis functions evaluated at grid points" .format(
                    self.__class__), UserWarning, stacklevel=2)
                # evaluate the lagrange basis in one 1d for both x and y at nodal points
                edge_basis_1d = [edge_basis(
                    self._face_nodes[i], self._face_nodes[i]) for i in range(2)]
                # store domain
                self.xi, self.eta = np.meshgrid(self._face_nodes[0], self._face_nodes[1])
            else:
                # evaluate the lagrange basis in one 1d for both x and y at quad points
                edge_basis_1d = [edge_basis(
                    self._face_nodes[i], self._quad_nodes[i]) for i in range(2)]
                # stor domain
                self.xi, self.eta = np.meshgrid(self._quad_nodes[0], self._quad_nodes[1])
            # tensor product
        else:
            self.xi, self.eta = np.meshgrid(*domain)
            edge_basis_1d = [edge_basis(
                self._face_nodes[i], domain[i]) for i in range(2)]
        self.basis = np.kron(edge_basis_1d[0], edge_basis_1d[1])

    def inner(self, other):
        self._inner_consistency_check(other)
        self.xi, self.eta = np.meshgrid(self._quad_nodes[0], self._quad_nodes[1])
        quad_weights_2d = np.kron(self._quad_weights[0], self._quad_weights[1]).reshape(
            np.size(self._quad_weights[0]) * np.size(self._quad_weights[1]), 1)

        g = self.mesh.g(self.xi.ravel('F'), self.eta.ravel('F'))

        M_2 = np.tensordot(
            self.basis, other.basis[:, :, np.newaxis] * (np.reciprocal(g) * quad_weights_2d), axes=((1), (1)))
        return M_2

    def wedged(self, other):
        """Integrate the wedge product of 2-form and a 0-form.

        The two forms is the trial function (on the columns) and the 0-form is the test function.

        """
        assert self.function_space.k + \
            other.function_space.k == self.mesh.dim, 'k-form wedge l-form, k+l should be equal to n'

        p_0 = np.max([self.p[0], other.p[0]])
        p_1 = np.max([self.p[1], other.p[1]])
        self.quad_grid = ('gauss', 'gauss'), (p_0, p_1)
        other.quad_grid = ('gauss', 'gauss'), (p_0, p_1)

        quad_weights_2d = np.kron(self._quad_weights[0], self._quad_weights[1]
                                  ).reshape(1,
                                            np.size(self._quad_weights[0]) * np.size(self._quad_weights[1]))
        W = np.tensordot(other.basis, self.basis * quad_weights_2d, axes=((1), (1)))
        return W


class LobattoFace(BasisForm_2):
    """Defines and element with two forms basis functions based on the Lobatto discretization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.face_grid = 'lobatto', 'lobatto'

    @BasisForm_2.face_grid.setter
    def face_grid(self, face_grid):
        self._face_grid = face_grid
        self._face_nodes = [getattr(quadrature, self._face_grid[i] +
                                    '_quad')(self.p[i])[0] for i in range(2)]


class GaussFace(BasisForm_2):
    """Defines and element with two forms basis functions based on the Gauss discretization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.face_grid = 'gauss', 'gauss'

    @BasisForm_2.face_grid.setter
    def face_grid(self, face_grid):
        self._face_grid = face_grid
        self._face_nodes = [getattr(quadrature, self._face_grid[i] +
                                    '_quad')(self.p[i])[0] for i in range(2)]


class ExtGaussFace(BasisForm_2):
    """Defines and element with two forms basis functions based on the extended_Gauss discretization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.face_grid = 'extended_gauss', 'extended_gauss'

    @BasisForm_2.face_grid.setter
    def face_grid(self, face_grid):
        self._face_grid = face_grid
        self._face_nodes = [getattr(quadrature, self._face_grid[i] +
                                    '_quad')(self.p[i])[0] for i in range(2)]


class FlexibleFace(BasisForm_2):
    """Defines and element with two forms basis functions based flexible dofs"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.face_grid = self.function_space.grid_type
        for i, grid in enumerate(self.face_grid):
            if grid == 'extended_gauss':
                self.num_basis *= int((self.p[i] + 2) / (self.p[i]))

    @BasisForm_2.face_grid.setter
    def face_grid(self, face_grid):
        self._face_grid = face_grid
        self._face_nodes = [getattr(quadrature, self._face_grid[i] +
                                    '_quad')(self.p[i])[0] for i in range(2)]


def BasisForm(function_space):
    """Wrapper around the elements."""
    elem_type = function_space.str_to_elem[function_space.form_type]
    return getattr(sys.modules[__name__], elem_type)(function_space)


if __name__ == '__main__':
    #    p = 3
    # grid_type = 'gauss', 'gauss'
    #    mesh = CrazyMesh(2, (1, 1), ((-1, 1), (-1, 1)), curvature=0.3)

    #    nx, ny = 1, 1
    mesh = CrazyMesh(2, (1, 1), ((-1, 1), (-1, 1)), curvature=0.3)
    fs = FunctionSpace(mesh, '0-gauss', 0)
    f0 = BasisForm(fs)
#    bf1.quad_grid = 'gauss', 0
#    print(bf1.inner(bf1))


#    function_space_l = FunctionSpace(mesh, '1-lobatto', 2)
#    basis_lobatto = BasisForm(function_space_l)
#
#    # # ------ example 1 0 - forms - ---------
#    # function_space = FunctionSpace(mesh, '1-lobatto', p)
#
#    # basis_lobatto = BasisForm(function_space)
#    # # form_0 = Function(function_space, continous=my_func, cochain=my_cochain)
#    # # print("p ", basis_lobatto.p)
#    # # print("nodes ", basis_lobatto._nodal_nodes)
#    # basis_lobatto.quad_grid = None, 2 * p
#    # # print('quad nodes ', basis_lobatto._quad_nodes)
#    # # basis_lobatto.evaluate_basis()
#    # # bug if the basis not evaluated at quad pts and just after that call inner product
#    # # print(basis_lobatto.basis)
#
#    # ------ example 2 0-forms ---------
#
#    # function_space = FunctionSpace(mesh, '0', p, grid_type)
#    # basis_flex = FlexibleNodal(function_space)
#    # basis_flex.quad_grid = ('lobatto', 'lobatto'), (4, 6)
#    # print(basis_flex.nodal_grid)
#    # M_0 = inner(basis_flex, basis_flex)
#    # print(np.shape(M_0))
#
#    # ------- example 3 1-forms
#    p = 2, 2
#    nx, ny = 1, 1
#    mesh = CrazyMesh(2, (nx, ny), ((0, 1), (0, 1)))
#
#    function_space = FunctionSpace(mesh, '1-lobatto', p)
#
#    basis_lobatto = LobattoEdge(function_space)
#    K = MeshFunction(mesh)
#    K.continous_tensor = [diff_tens_11, diff_tens_12, diff_tens_22]
##    p = 2, 2
##    nx, ny = 2, 2
##    mesh = CrazyMesh(2, (nx, ny), ((0, 1), (0, 1)),0.1)
##    function_space = FunctionSpace(mesh, '1-ExtGauss', p)
##    basis_ExtGau = ExtGaussEdge(function_space)
##
##
##    K = MeshFunction(mesh)
##    K.continous_tensor = [diff_tens_11, diff_tens_12, diff_tens_22]
# print(basis_lobatto._nodal_nodes)
# print(basis_lobatto._edge_nodes)
# print(basis_lobatto.quad_grid)
# print(basis_lobatto.basis)
#
#    basis_lobatto.quad_grid = 'gauss'
#    # basis_lobatto.evaluate_basis()
#
#    # print(np.shape(basis_lobatto.basis))
#    t0 = time.time()
#    M_0 = inner(basis_lobatto, basis_lobatto)
#    print(M_0)
#    t1 = time.time()
#    print("time : ", t1 - t0)

    # ------- example 4 2-forms
    # p = 20, 20
    # nx, ny = 100, 100
    # mesh = CrazyMesh(2, (nx, ny), ((-1, 1), (-1, 1)))
    # function_space = FunctionSpace(mesh, '2', p)
    # basis_lobatto = LobattoFace(function_space)
    # # print("p ", basis_lobatto.p)
    # # print("nodes ", basis_lobatto._face_nodes)
    # basis_lobatto.quad_grid = ('lobatto', 'lobatto'), (20, 20)
    # # print('quad nodes ', basis_lobatto._quad_nodes)
    # # basis_lobatto.evaluate_basis()
    # # # bug if the basis not evaluated at quad pts and just after that call inner product
    # # print(basis_lobatto.basis)
    # M_2 = inner(basis_lobatto, basis_lobatto)
    # print(np.shape(M_2))

    # ------- example Wrapper
    # function_space = FunctionSpace(mesh, '2-lobatto', 2)
    # basis = BasisFunction(function_space)
    # print(type(basis))~Z~Z`zx
