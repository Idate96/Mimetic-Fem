"""Series of classes for mesh generation.
"""
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

class AbstractMesh(ABC):
    """Abstract class serving as template for the generation of any mesh type.

    Attributes:
    ----------
    'num_elements' :    tuple(int, size : 2)
                        tuple containing the number of elements
                        in the x and y direction, ie num_elements=(nx,ny)
    Methods:
    -------
    'mapping'
    'dx_dxi'
    'dx_deta'
    'dy_dxi'
    'dy_deta'

    :First Added:   2017-05-10

    Revisions:
    ---------
    """
    def __init__(self, dim, num_elements):
        self.num_elements = num_elements
        self.dim = dim

    @abstractmethod
    def mapping(element, eta, xi):
        pass

    @abstractmethod
    def dx_dxi(element, eta, xi):
        pass

    @abstractmethod
    def dx_deta(element, eta, xi):
        pass

    @abstractmethod
    def dy_dxi(element, eta, xi):
        pass

    @abstractmethod
    def dy_deta(element, eta, xi):
        pass

class CrazyMesh(AbstractMesh):
    """
    CrazyMesh class generates a tensor product mesh. The mesh can be deformed
    through a non linear mapping.

    Args:
    ----
        dim :   int
                dim is the dimension of the manifold
                # currenly implemente just in 2D

        num_elements :  tuple (int, size : 2)
                        num_elements specifies the number of elements in the x
                        and y dimension.
                        example : num_elements = (i,j) implements i elements in x
                        and j elements in y direction in the mesh

        bounds_domain : tuple (float64, size : 4)
                        bounds_domain specifies the nodal coordinates at the corners
                        of the domain. The first entry is a tuple containing the
                        coordinates of the extreme points in the x dimension, second entry
                        the cooridinates in the y dimension
                        example : bounds_domain = ((a_x,b_x),(a_y,b_y)) where
                        b_x > a_x, b_y > a_y

        curvature :     float64 (optional)
                        specifies the curvature to be applied to the domain.
                        Typical range [0-0.2]

        nodes :         tuple()



    """

    def __init__(self, dim, elements_layout, bounds_domain, curvature=0, nodes=None):
        # nodes possible confilict with bounds_domain
        """num_elements (nx,ny)"""
        super().__init__(dim, elements_layout[0] * elements_layout[1])
        self.n_x, self.n_y = elements_layout
        self.n = elements_layout
        self.curvature = curvature
        # bounds_domain = ((xa,xb),(ya,yb))
        self.x_bounds, self.y_bounds = bounds_domain

        # TODO: give options for unqually spaced elements
        self.nodes = nodes
        self.bounds_x_elements, self.bounds_y_elements = None, None
        self.curvature = curvature

    @property
    def n_x(self):
        return self._n_x

    @n_x.setter
    def n_x(self, n_x: int):
        try:
            assert n_x > 0
            self._n_x = n_x
        except Exception as t:
            raise ValueError("The number of elements should be positive")

    @property
    def n_y(self):
        return self._n_y

    @n_y.setter
    def n_y(self, n_y: int):
        try:
            assert n_y > 0
            self._n_y = n_y
        except Exception as t:
            raise ValueError("The number of elements should be positive")

    def set_elements_bounds(self):
        x_el_start = [(self.x_bounds[1] - self.x_bounds[0]) /
                      self.n_x * i + self.x_bounds[0] for i in range(self.n_x)]
        y_el_start = [(self.y_bounds[1] - self.y_bounds[0]) /
                      self.n_y * i + self.y_bounds[0] for i in range(self.n_y)]
        x_el_end = [(self.x_bounds[1] - self.x_bounds[0]) /
                    self.n_x * i + x_el_start[1] for i in range(self.n_x)]
        y_el_end = [(self.y_bounds[1] - self.y_bounds[0]) /
                    self.n_y * i + y_el_start[1] for i in range(self.n_y)]
        self.x_el_bounds = np.asarray(list(zip(x_el_start, x_el_end)))
        self.y_el_bounds = np.asarray(list(zip(y_el_start, y_el_end)))

    def mapping(self, xi, eta, element=None):
        if not isinstance(element, int):
            if element is None:
                element = np.arange(self.num_elements)
            x, y = np.zeros((*np.shape(xi), np.size(element))
                            ), np.zeros((*np.shape(eta), np.size(element)))
            for i in range(np.size(element)):
                x[..., i], y[..., i] = self.mapping(xi, eta, element=i)
            return x, y

        assert element < self.num_elements, "Element number out of bounds"
        # the logical spacing of the elements
        delta_x = 2.0 / self.n_x
        delta_y = 2.0 / self.n_y

        # x and y indices of the element
        index_x = np.ceil((element + 1) / self.n_y) - 1
        index_y = (element) % self.n_y

        x_left = -1 + delta_x * index_x
        y_left = -1 + delta_y * index_y

        x = (((xi + 1) * 0.5 * delta_x) + x_left + self.curvature * np.sin(np.pi * ((eta + 1) * 0.5 * delta_y + y_left)) *
             np.sin(np.pi * ((xi + 1) * 0.5 * delta_x + x_left)) + 1) * (self.x_bounds[1] - self.x_bounds[0]) * 0.5 + self.x_bounds[0]

        y = (((eta + 1) * 0.5 * delta_y) + y_left + self.curvature * np.sin(np.pi * ((eta + 1) * 0.5 * delta_y + y_left)) *
             np.sin(np.pi * ((xi + 1) * 0.5 * delta_x + x_left)) + 1) * (self.y_bounds[1] - self.y_bounds[0]) * 0.5 + self.y_bounds[0]

        return x, y

    def dx_dxi(self, xi, eta, element=None):
        if not isinstance(element, int):
            if element is None:
                element = np.arange(self.num_elements)
            dx_dxi_result = np.zeros((*np.shape(xi), np.size(element)))
            for i in range(np.size(element)):
                dx_dxi_result[..., i] = self.dx_dxi(xi, eta, element=i)
            return dx_dxi_result

        assert element <= self.num_elements, "Element number out of bounds"

        delta_x = 2.0 / self.n_x
        delta_y = 2.0 / self.n_y

        # x and y indices of the element
        index_x = np.ceil((element + 1) / self.n_y) - 1
        index_y = (element) % self.n_y

        x_left = -1 + delta_x * index_x
        y_left = -1 + delta_y * index_y
        # print('self.x_bounds[1] - self.x_bounds[0]',
        #   self.x_bounds[1] - self.x_bounds[0])

        dx_dxi_result = 0.5 * delta_x * \
            (self.x_bounds[1] - self.x_bounds[0]) * 0.5 + np.pi * delta_x * 0.5 * self.curvature * np.sin(np.pi * ((eta + 1) * 0.5 *
                                                                                                                   delta_y + y_left)) * np.cos(np.pi * ((xi + 1) * 0.5 * delta_x + x_left)) * (self.x_bounds[1] - self.x_bounds[0]) * 0.5

        return dx_dxi_result

    def dx_deta(self, xi, eta, element=None):

        if not isinstance(element, int):
            if element is None:
                element = np.arange(self.num_elements)
            dx_deta_result = np.zeros((*np.shape(xi), np.size(element)))
            for i in range(np.size(element)):
                dx_deta_result[..., i] = self.dx_deta(xi, eta, element=i)
            return dx_deta_result

        assert element <= self.num_elements, "Element number out of bounds"

        delta_x = 2.0 / self.n_x
        delta_y = 2.0 / self.n_y

        # x and y indices of the element
        index_x = np.ceil((element + 1) / self.n_y) - 1
        index_y = (element) % self.n_y

        x_left = -1 + delta_x * index_x
        y_left = -1 + delta_y * index_y

        dx_deta_result = np.pi * delta_y * 0.5 * self.curvature * \
            np.cos(np.pi * ((eta + 1) * 0.5 * delta_y + y_left)) * \
            np.sin(np.pi * ((xi + 1) * 0.5 * delta_x + x_left)) * \
            (self.x_bounds[1] - self.x_bounds[0]) * 0.5

        return dx_deta_result

    def dy_dxi(self, xi, eta, element=None):
        if not isinstance(element, int):

            if element is None:
                element = np.arange(self.num_elements)

            dy_dxi_result = np.zeros((*np.shape(xi), np.size(element)))
            for i in range(np.size(element)):
                dy_dxi_result[..., i] = self.dy_dxi(xi, eta, element=i)
            return dy_dxi_result

        assert element <= self.num_elements, "Element number out of bounds"

        delta_x = 2.0 / self.n_x
        delta_y = 2.0 / self.n_y

        # x and y indices of the element
        index_x = np.ceil((element + 1) / self.n_y) - 1
        index_y = (element) % self.n_y

        x_left = -1 + delta_x * index_x
        y_left = -1 + delta_y * index_y

        dy_dxi_result = np.pi * delta_x * 0.5 * self.curvature * \
            np.sin(np.pi * ((eta + 1) * 0.5 * delta_y + y_left)) * \
            np.cos(np.pi * ((xi + 1) * 0.5 * delta_x + x_left)) * \
            (self.y_bounds[1] - self.y_bounds[0]) * 0.5

        return dy_dxi_result

    def dy_deta(self, xi, eta, element=None):
        if not isinstance(element, int):
            if element is None:
                element = np.arange(self.num_elements)
            dy_deta_result = np.zeros((*np.shape(xi), np.size(element)))
            for i in range(np.size(element)):
                dy_deta_result[..., i] = self.dy_deta(xi, eta, element=i)
            return dy_deta_result

        assert element <= self.num_elements, "Element number out of bounds"

        delta_x = 2.0 / self.n_x
        delta_y = 2.0 / self.n_y

        # x and y indices of the element
        index_x = np.ceil((element + 1) / self.n_y) - 1
        index_y = (element) % self.n_y

        x_left = -1 + delta_x * index_x
        y_left = -1 + delta_y * index_y

        dy_deta_result = 0.5 * delta_y * (self.y_bounds[1] - self.y_bounds[0]) * 0.5 + \
            np.pi * delta_y * 0.5 * self.curvature * \
            np.cos(np.pi * ((eta + 1) * 0.5 * delta_y + y_left)) * \
            np.sin(np.pi * ((xi + 1) * 0.5 * delta_x + x_left)) * \
            (self.y_bounds[1] - self.y_bounds[0]) * 0.5
        return dy_deta_result

    def g(self, xi, eta, element=None):
        metric_term = self.dx_dxi(xi, eta, element) * self.dy_deta(xi, eta, element) - \
            self.dx_deta(xi, eta, element) * self.dy_dxi(xi, eta, element)
        return np.abs(metric_term)

    def g_11(self, xi, eta, element=None):
        g_11_result = (self.dx_deta(xi, eta, element) * self.dx_deta(xi, eta, element) +
                       self.dy_deta(xi, eta, element) * self.dy_deta(xi, eta, element)) / self.g(xi, eta, element)**2
        return g_11_result

    def g_12(self, xi, eta, element=None):
        g_12_result = -(self.dy_deta(xi, eta, element) * self.dy_dxi(xi, eta, element) +
                        self.dx_dxi(xi, eta, element) * self.dx_deta(xi, eta, element)) / self.g(xi, eta, element)**2
        return g_12_result

    def g_22(self, xi, eta, element=None):
        g_22_result = (self.dx_dxi(xi, eta, element) * self.dx_dxi(xi, eta, element) +
                       self.dy_dxi(xi, eta, element) * self.dy_dxi(xi, eta, element)) / self.g(xi, eta, element)**2
        return g_22_result
    
    def J11(self, xi, eta, element = None):
        return self.dx_dxi(xi, eta, element)
    
    def J12(self, xi, eta, element = None):
        return self.dx_deta(xi, eta, element)
    
    def J21(self, xi, eta, element = None):
        return self.dy_dxi(xi, eta, element)
    
    def J22(self, xi, eta, element = None):
        return self.dy_deta(xi, eta, element)
    
    def detJ(self, xi, eta, element = None):
        return self.g(xi,eta,element)
    
    def invJ11(self, xi, eta, element = None):
        return ( 1/self.g(xi,eta,element) ) * self.dy_deta(xi, eta, element)
    
    def invJ12(self, xi, eta, element = None):
        return -( 1/self.g(xi,eta,element) ) * self.dx_deta(xi, eta, element)
    
    def invJ21(self, xi, eta, element = None):
        return -( 1/self.g(xi,eta,element) ) * self.dy_dxi(xi, eta, element)
    
    def invJ22(self, xi, eta, element = None):
        return ( 1/self.g(xi,eta,element) ) * self.dx_dxi(xi, eta, element)
    
    def detinvJ(self, xi, eta, element = None):
        return 1/self.g(xi,eta,element)

    def metric_tensor(self, xi, eta, element=None):
        """Calculate all the components of the metric tensor."""
        dx_deta = self.dx_deta(xi, eta, element)
        dx_dxi = self.dx_dxi(xi, eta, element)
        dy_deta = self.dy_deta(xi, eta, element)
        dy_dxi = self.dy_dxi(xi, eta, element)
        g = (dx_dxi * dy_deta -
             dx_deta * dy_dxi)
        g_11 = (dx_deta * dx_deta +
                dy_deta * dy_deta) / g
        g_12 = -(dy_deta * dy_dxi +
                 dx_dxi * dx_deta) / g
        g_22 = (dx_dxi * dx_dxi +
                dy_dxi * dy_dxi) / g
        return g_11, g_12, g_22

if __name__ == '__main__':
    mesh = CrazyMesh(2, (2, 2), ((-1, 1), (-1, 1)), curvature=0.1)
    xi = np.linspace(-1, 1, 10)
    xi, eta = np.meshgrid(xi, xi)
    x, y = mesh.mapping(xi, eta, 0)
    plt.plot(x, y, '-x')
    plt.show()
