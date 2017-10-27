import numpy as np
from quadrature import lobatto_quad
import matplotlib.pyplot as plt
from functools import lru_cache


def derivative_poly_nodes(p, nodes):
    """For computation of the derivative at the nodes a more efficient and accurate formula can
       be used, see [1]:

                 | \frac{c_{k}}{c_{j}}\frac{1}{x_{k}-x_{j}},          k \neq j
                 |
       d_{kj} = <
                 | \sum_{l=1,l\neq k}^{p+1}\frac{1}{x_{k}-x_{l}},     k = j
                 |

                 with
        c_{k} = \prod_{l=1,l\neq k}^{p+1} (x_{k}-x_{l}).

    Args:
        p (int) = degree of polynomial
        type_poly (string) = 'Lobatto', 'Gauss', 'Extended Gauss'
    [1] Costa, B., Don, W. S.: On the computation of high order
      pseudospectral derivatives, Applied Numerical Mathematics, vol.33
       (1-4), pp. 151-159
        """
    # compute distances between the nodes
    xi_xj = nodes.reshape(p + 1, 1) - nodes.reshape(1, p + 1)
    # diagonals to one
    xi_xj[np.diag_indices(p + 1)] = 1
    # compute (ci's)
    c_i = np.prod(xi_xj, axis=1)
    # compute ci/cj = ci_cj(i,j)
    c_i_div_cj = np.transpose(c_i.reshape(1, p + 1) / c_i.reshape(p + 1, 1))
    # result formula
    derivative = c_i_div_cj / xi_xj
    # put the diagonals equal to zeros
    derivative[np.diag_indices(p + 1)] = 0
    # compute the diagonal values enforning sum over rows = 0
    derivative[np.diag_indices(p + 1)] = -np.sum(derivative, axis=1)
    return derivative


def lagrange_basis(nodes, x=None):
    if x is None:
        x = nodes
    p = np.size(nodes)
    basis = np.ones((p, np.size(x)))
    # lagrange basis functions
    for i in range(p):
        for j in range(p):
            if i != j:
                basis[i, :] *= (x - nodes[j]) / (nodes[i] - nodes[j])
    return basis

def derivative_poly(p, nodes, x):
    """Return the derivatives of the polynomials in the domain x."""
    nodal_derivative = derivative_poly_nodes(p, nodes)
    polynomials = lagrange_basis(nodes, x)
    return np.transpose(nodal_derivative) @ polynomials

def edge_basis(nodes, x=None):
    """Return the edge polynomials."""
    if x is None:
        x = nodes
    p = np.size(nodes) - 1
    derivatives_poly = derivative_poly(p, nodes, x)
    edge_poly = np.zeros((p, np.size(x)))
    for i in range(p):
        for j in range(i + 1):
            edge_poly[i] -= derivatives_poly[j, :]
    return edge_poly


# if __name__ == '__main__':
#     p = 4
#     nodes, weights = lobatto_quad(p)
#     x = np.linspace(-1, 1, 50)
#
#     l_basis = lagrange_basis(nodes, x)
#     e_basis = edge_basis(nodes, x)
#
#     plt.plot(x, np.transpose(e_basis))
#     plt.show()
