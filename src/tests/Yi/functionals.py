# -*- coding: utf-8 -*-
"""
@author: Yi Zhang. Created on June 3 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import numpy as np
from functools import partial
from scipy.special import legendre
import types

# %% --------------------------------------------------------------------------
def mbfv(x, p, poly_type):
    assert np.min(x) >= -1 and np.max(x) <=1, "x should be in [-1,1]"
    assert p >= 1 and p%1 == 0, "p value is wrong, should be positve integer"

    if poly_type == "LobN": # lobatto polynomials
        nodes, weights = lobatto_quad(p)
        basis=lagrange_basis(nodes, x)
        return  basis
    elif poly_type == "LobE": # lobatto edges functions
        nodes, weights = lobatto_quad(p)
        basis=edge_basis(nodes, x)
        return basis

    elif poly_type == "GauN":  # gauss polynomials
        nodes, weights = gauss_quad(p)
        basis=lagrange_basis(nodes, x)
        return basis
    elif poly_type == "GauE":  # gauss edges functions
        nodes, weights = gauss_quad(p)
        basis=edge_basis(nodes, x)
        return basis

    elif poly_type == "etGN": # extended-gauss polynomials
        nodes, weights = extended_gauss_quad(p)
        basis=lagrange_basis(nodes, x)
        return basis
    elif poly_type == "etGE": # extended-gauss edges functions
        nodes, weights = extended_gauss_quad(p)
        basis=edge_basis(nodes, x)
        return basis

    else:
        raise Exception("Error, poly_type wrong......")
        
# %% --------------------------------------------------------------------------
def _legendre_prime(x, n):
    """Calculate first derivative of the nth Legendre Polynomial recursively.

    Args:
        x (float,np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_p (np.array) = value first derivative of L_n.
    """
    # P'_n+1 = (2n+1) P_n + P'_n-1
    # where P'_0 = 0 and P'_1 = 1
    # source: http://www.physicspages.com/2011/03/12/legendre-polynomials-recurrence-relations-ode/
    if n == 0:
        if isinstance(x, np.ndarray):
            return np.zeros(len(x))
        elif isinstance(x, (int, float)):
            return 0
    if n == 1:
        if isinstance(x, np.ndarray):
            return np.ones(len(x))
        elif isinstance(x, (int, float)):
            return 1
    legendre_p = (n * legendre(n - 1)(x) - n * x * legendre(n)(x))/(1-x**2)
    return legendre_p

# %% --------------------------------------------------------------------------
def _legendre_prime_lobatto(x,n):
    return (1-x**2)**2*_legendre_prime(x,n)

# %% --------------------------------------------------------------------------
def _legendre_double_prime(x, n):
    """Calculate second derivative legendre polynomial recursively.

    Args:
        x (float,np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_pp (np.array) = value second derivative of L_n.
    """
    legendre_pp = 2 * x * _legendre_prime(x, n) - n * (n + 1) * legendre(n)(x)
    return legendre_pp * (1 - x ** 2)

# %% -------------------------------------------------------------------------- 
def _newton_method(f, dfdx, x_0, n_max, min_error=np.finfo(float).eps * 10):
    """Newton method for rootfinding.

    It garantees quadratic convergence given f'(root) != 0 and abs(f'(ξ)) < 1
    over the domain considered.

    Args:
        f (obj func) = function
        dfdx (obj func) = derivative of f
        x_0 (float) = starting point
        n_max (int) = max number of iterations
        min_error (float) = min allowed error

    Returns:
        x[-1] (float) = root of f
        x (np.array) = history of convergence
    """
    x = [x_0]
    for i in range(n_max - 1):
        x.append(x[i] - f(x[i]) / dfdx(x[i]))
        if abs(x[i + 1] - x[i]) < min_error:
            return x[-1]
    print('WARNING : Newton did not converge to machine precision \nRelative error : ',
          x[-1] - x[-2])
    return x[-1]

# %% --------------------------------------------------------------------------
def lobatto_quad(p):
    """Gauss Lobatto quadrature.

    Args:
        p (int) = order of quadrature

    Returns:
        nodal_pts (np.array) = nodal points of quadrature
        w (np.array) = correspodent weights of the quarature.
    """
    # nodes
    x_0 = np.cos(np.arange(1, p) / p * np.pi)
    nodal_pts = np.zeros((p + 1))
    # final and inital pt
    nodal_pts[0] = 1
    nodal_pts[-1] = -1
    # Newton method for root finding
    for i, ch_pt in enumerate(x_0):
        leg_p = partial(_legendre_prime_lobatto, n=p)
        leg_pp = partial(_legendre_double_prime, n=p)
        nodal_pts[i + 1] = _newton_method(leg_p, leg_pp, ch_pt, 100)

    # weights
    weights = 2 / (p * (p + 1) * (legendre(p)(nodal_pts))**2)

    return nodal_pts[::-1], weights

# %% --------------------------------------------------------------------------
def gauss_quad(p):
    # Chebychev pts as inital guess
    x_0 = np.cos(np.arange(1, p + 1) / (p + 1) * np.pi)
    nodal_pts = np.empty(p)
    for i, ch_pt in enumerate(x_0):
        leg = legendre(p)
        leg_p = partial(_legendre_prime, n=p)
        nodal_pts[i] = _newton_method(leg, leg_p, ch_pt, 100)

    weights = 2 / (p * legendre(p - 1)(nodal_pts)
                   * _legendre_prime(nodal_pts, p))
    return nodal_pts[::-1], weights

# %% --------------------------------------------------------------------------
def extended_gauss_quad(p):
    nodes, weights = gauss_quad(p )
    ext_nodes = np.ones((p + 2))
    ext_nodes[0] = -1
    ext_nodes[1:-1] = nodes
    ext_weights = np.zeros(p + 2)
    ext_weights[1:-1] = weights
    return ext_nodes, ext_weights

# %% --------------------------------------------------------------------------
def _derivative_poly_nodes(p, nodes):
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

# %% --------------------------------------------------------------------------
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

# %% --------------------------------------------------------------------------
def derivative_poly(p, nodes, x):
    """Return the derivatives of the polynomials in the domain x."""
    nodal_derivative = _derivative_poly_nodes(p, nodes)
    polynomials = lagrange_basis(nodes, x)
    return np.transpose(nodal_derivative) @ polynomials

# %% --------------------------------------------------------------------------
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

# %% --------------------------------------------------------------------------
def _size_check(basis):
    poly_type, p = basis
    if poly_type == 'lobatto_node':
        return p+1
    elif poly_type == 'lobatto_edge':
        return p
    elif poly_type == 'gauss_node':
        return p
    elif poly_type == 'gauss_edge':
        return p-1
    elif poly_type == 'ext_gauss_node':
        return p+2
    elif poly_type == 'ext_gauss_edge':
        return p+1
    else:
        raise Exception("mimetic basis function type wrong......")
        
# %% --------------------------------------------------------------------------
def _bf_value( basis, x):
    poly_type, p = basis
    if poly_type == 'lobatto_node':
        bfv = mbfv(x, p, 'LobN')
        return bfv
    elif poly_type == 'lobatto_edge':
        bfv = mbfv(x, p, "LobE")
        return bfv

    elif poly_type == 'gauss_node':
        bfv = mbfv(x, p, "GauN")
        return bfv
    elif poly_type == 'gauss_edge':
        bfv = mbfv(x, p, "GauE")
        return bfv

    elif poly_type == 'ext_gauss_node':
        bfv = mbfv(x, p, "etGN")
        return bfv
    elif poly_type == 'ext_gauss_edge':
        bfv = mbfv(x, p, "etGE")
        return bfv
    else:
        raise Exception("mimetic basis function type wrong......")
        
# %% --------------------------------------------------------------------------
def integral0d_(metric, basis_1, Quad):
    # using basis_1 to test basis_2
    if isinstance(metric, types.FunctionType):
        pass
    elif isinstance(metric, int) or isinstance(metric, float):
        temp=metric
        def fun(x):
            return temp
        metric=fun
    else:
        raise Exception("metric type wrong, only accept function, int or float")

    sd1 = _size_check(basis_1)
    
    QuadType, QuadOrder = Quad
    if QuadOrder <= 0:
        QuadOrder = np.ceil((sd1)/2 + 10)

    if QuadType == 'gauss':
        Qnodes, weights = gauss_quad(QuadOrder)
    elif QuadType == 'lobatto':
        Qnodes, weights = lobatto_quad(QuadOrder)
    else:
        raise Exception("Quad Type should be gauss or lobatto.......")
    
    basis_1 = _bf_value( basis_1, Qnodes)
    
    metric = metric(Qnodes)
    if np.size(metric) == 1:
        metric = metric * np.ones((np.size(Qnodes)))
    
    IntValue = np.einsum('ik,k,k->i', basis_1, metric, weights)
    return IntValue

# %% --------------------------------------------------------------------------
def integral1d_(metric, basis_1, basis_2, Quad):
    # using basis_1 to test basis_2
    if isinstance(metric, types.FunctionType):
        pass
    elif isinstance(metric, int) or isinstance(metric, float):
        temp=metric
        def fun(x):
            return temp
        metric=fun
    else:
        raise Exception("metric type wrong, only accept function, int or float")

    sd1 = _size_check(basis_1)
    sd2 = _size_check(basis_2)
    
    QuadType, QuadOrder = Quad
    if QuadOrder <= 0:
        QuadOrder = np.ceil((sd1 + sd2)/2 + 10)

    if QuadType == 'gauss':
        Qnodes, weights = gauss_quad(QuadOrder)
    elif QuadType == 'lobatto':
        Qnodes, weights = lobatto_quad(QuadOrder)
    else:
        raise Exception("Quad Type should be gauss or lobatto.......")
    
    basis_1 = _bf_value( basis_1, Qnodes)
    basis_2 = _bf_value( basis_2, Qnodes)
    
    metric = metric(Qnodes)
    if np.size(metric) == 1:
        metric = metric * np.ones((np.size(Qnodes)))
    
    IntValue = np.einsum('ik,jk,k,k->ij', basis_1, basis_2, metric, weights)
    
    return IntValue

# %% --------------------------------------------------------------------------
if __name__ == '__main__':
    IntValue1 = integral1d_(1, ('lobatto_edge',3), ('gauss_node',3), ('gauss',50))
    print(IntValue1)
    
    IntValue0 = integral0d_(1, ('lobatto_edge',3), ('gauss',50))
    print(IntValue0)
