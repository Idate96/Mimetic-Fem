import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from functools import partial
from src.legendre_functions import legendre_prime, legendre_double_prime, legendre_prime_lobatto
from scipy.special import legendre

def newton_method(f, dfdx, x_0, n_max, min_error=np.finfo(float).eps * 10):
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

# %%
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
        leg_p = partial(legendre_prime_lobatto, n=p)
        leg_pp = partial(legendre_double_prime, n=p)
        nodal_pts[i + 1] = newton_method(leg_p, leg_pp, ch_pt, 100)

    # weights
    weights = 2 / (p * (p + 1) * (legendre(p)(nodal_pts))**2)

    return nodal_pts[::-1], weights


def gauss_quad(p):
    p += 1
    # Chebychev pts as inital guess
    x_0 = np.cos(np.arange(1, p + 1) / (p + 1) * np.pi)
    nodal_pts = np.empty(p)
    for i, ch_pt in enumerate(x_0):
        leg = legendre(p)
        leg_p = partial(legendre_prime, n=p)
        nodal_pts[i] = newton_method(leg, leg_p, ch_pt, 100)

    weights = 2 / (p * legendre(p - 1)(nodal_pts)
                   * legendre_prime(nodal_pts, p))
    return nodal_pts[::-1], weights


# def quad_weights_2d(quad_weight_x, quad_weights_y):
#     return np.kron(quad_weight_x, quad_weights_y)


def extended_gauss_quad(p):
    nodes, weights = gauss_quad(p)
    ext_nodes = np.ones((p + 3))
    ext_nodes[0] = -1
    ext_nodes[1:-1] = nodes
    ext_weights = np.zeros(p + 3)
    ext_weights[1:-1] = weights
    return ext_nodes, ext_weights


if __name__ == '__main__':
    p = 3
    nodes, weights = gauss_quad(p)
    nodes_ext = extended_gauss_quad(p)[0]
    print(nodes, nodes_ext)
    print(weights)