from scipy.special import legendre
import numpy as np

def legendre_prime(x, n):
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

def legendre_prime_lobatto(x,n):
    return (1-x**2)**2*legendre_prime(x,n)

def legendre_double_prime(x, n):
    """Calculate second derivative legendre polynomial recursively.

    Args:
        x (float,np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_pp (np.array) = value second derivative of L_n.
    """
    legendre_pp = 2 * x * legendre_prime(x, n) - n * (n + 1) * legendre(n)(x)
    return legendre_pp * (1 - x ** 2)
