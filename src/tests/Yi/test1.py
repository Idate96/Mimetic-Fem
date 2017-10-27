import polynomials
import quadrature
import numpy as np
import matplotlib.pyplot as plt

p = 5

#nodes, weights = quadrature.extended_gauss_quad(p)
#nodes, weights = quadrature.gauss_quad(p)
#nodes, weights = quadrature.lobatto_quad(p)

nodes = [-1, -0.9, 0.5, 1]
x = np.linspace(-1,1,200)

a = polynomials.lagrange_basis(nodes, x)
#a = polynomials.edge_basis(nodes, x)

plt.figure
for i in range(4):
    plt.plot(x, a[i,:])
plt.show