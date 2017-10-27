# -*- coding: utf-8 -*-
"""
(SHORT NAME EXPLANATION)
"""
#import path_magic
from function_space import FunctionSpace
import numpy as np
from mesh import CrazyMesh
from forms import Form
from hodge import hodge
from coboundaries import d
from assemble import assemble
from _assembling import assemble_
import matplotlib.pyplot as plt
from quadrature import extended_gauss_quad
from scipy.integrate import quad
from sympy import Matrix
import scipy.io
from scipy import sparse
import scipy as sp

# %% exact solution define
# u^{(1)} = { u,  v }^T
def u(x,y):
	return   +np.cos(np.pi*x) * np.sin(np.pi*y)

def v(x,y):
	return   -np.sin(np.pi*x) * np.cos(np.pi*y)

# %% define the mesh
px = py = 12
nx = ny = 1
mesh = CrazyMesh( 2, (nx, ny), ((-1, 1), (-1, 1)), 0.0 )
xi = eta = np.linspace( -1, 1, np.ceil(100 / (nx * ny)) + 1 )

# %% define function space
func_space_gl0 = FunctionSpace(mesh, '0-lobatto', (px + 1, py + 1), is_inner=False)
func_space_gl1 = FunctionSpace(mesh, '1-lobatto', (px + 1, py + 1), is_inner=False)
func_space_gl2 = FunctionSpace(mesh, '2-lobatto', (px + 1, py + 1), is_inner=False)
func_space_eg0 = FunctionSpace(mesh, '0-ext_gauss', (px, py))
func_space_eg1 = FunctionSpace(mesh, '1-ext_gauss', (px, py))
func_space_eg2 = FunctionSpace(mesh, '2-ext_gauss', (px, py))

# %% matrices are going to be used
#egE21_assembled = egE21_assembled.todense()
#print(glE21_assembled)

# %% define the form
#glE21_assembled = assemble_(mesh, glE21, fo2.function_space.dof_map.dof_map,
#                             uo1.function_space.dof_map.dof_map, mode='replace')

# %% uo1
un1 = Form(func_space_gl1)
un1.discretize((v,u))
un1.reconstruct(xi, eta)
(x, y), data_dx, data_dy = un1.export_to_plot()
plt.contourf(x, y, data_dx)
plt.title('1.1). lobatto u^{(n-1)} dx')
plt.colorbar()
plt.show()

plt.contourf(x, y, data_dy)
plt.title('1.2). lobatto u^{(n-1)} dy')
plt.colorbar()
plt.show()

# %%
def d_21_lobatto_outer(p):
    px, py = p
    total_vol = px * py
    total_edges = px * (py + 1) + py * (px + 1)
    E21 = np.zeros((total_vol, total_edges))

    for i in range(px):
        for j in range(py):
            volij = i * py + j
            edge_bottom = i * (py + 1) + j
            edge_top = i * (py + 1) + j + 1
            edge_left = (py + 1) * px + i * py + j
            edge_right = (py + 1) * px + (i + 1) * py + j

            E21[volij, edge_bottom] = +1
            E21[volij, edge_top] = -1
            E21[volij, edge_left] = -1
            E21[volij, edge_right] = +1
    return E21
glE21 = d_21_lobatto_outer((px+1, py+1))
#glE21 = d(func_space_gl1)
glE21_assembled = assemble(glE21, (func_space_gl2, func_space_gl1))

fn = Form(func_space_gl2)
fn.cochain = glE21_assembled.dot(un1.cochain)
#fo2 = d(uo1)
#fn.reconstruct(xi, eta)
#(x, y), data = fn.export_to_plot()
#plt.contourf(x, y, data)
#plt.title('2). lobatto f^{(n)} dx \wedge dy')
#plt.colorbar()
#plt.show()

# %%
H02gl = hodge(func_space_gl2)
H02gl_assembled   = assemble(H02gl  , (func_space_eg0, func_space_gl2))

f0  = Form(func_space_eg0)


f0_cochain_internal = H02gl_assembled.dot(fn.cochain)
f0.cochain = np.concatenate((f0_cochain_internal, np.zeros(
        f0.function_space.num_dof - f0.basis.num_basis * f0.mesh.num_elements)), axis=0)
#f0.reconstruct(xi, eta)
#(x, y), data = f0.export_to_plot()
#plt.contourf(x, y, data)
#plt.title("3). ext_gauss \\tilde{f}^{(0)}")
#plt.colorbar()
#plt.show()

def f0_fun(x,y):
    return -2 * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)

f0_exact  = Form(func_space_eg0);
f0_exact.discretize(f0_fun)

def CrazyMesh_2d_extended_gauss0_general_boundary_nodes(mesh, p, gathering_matrix):
    p += 1
    nx = mesh.n_x
    ny = mesh.n_y

    Left = np.zeros(shape=(ny * p), dtype=np.int16)
    Right = np.zeros(shape=(ny * p), dtype=np.int16)
    Bottom = np.zeros(shape=(nx * p), dtype=np.int16)
    Top = np.zeros(shape=(nx * p), dtype=np.int16)

    for J in range(ny):
        eleidLeft = J
        Left[J * p: J * p + p] = gathering_matrix[eleidLeft, p**2: p**2 + p]

        eleidRight = (nx - 1) * ny + J
        Right[J * p: J * p + p] = gathering_matrix[eleidRight, p**2 + p: p**2 + 2 * p]

    for I in range(nx):

        eleidBottom = I * ny
        Bottom[I * p: I * p + p] = gathering_matrix[eleidBottom, p**2 + 2 * p: p**2 + 3 * p]

        eleidTop = I * ny + ny - 1
        Top[I * p: I * p + p] = gathering_matrix[eleidTop, p**2 + 3 * p: p**2 + 4 * p]

    return Left, Right, Bottom, Top


Left, Right, Bottom, Top = CrazyMesh_2d_extended_gauss0_general_boundary_nodes(
                                mesh, px, f0.function_space.dof_map.dof_map)
Boundarypoint = np.hstack((Left, Right, Bottom, Top))

for i in range(np.size(Boundarypoint)):
    f0.cochain[ Boundarypoint[i]] = f0_exact.cochain[Boundarypoint[i]]

# %%
def d_10_ext_gauss_inner(p):
    px, py = p
    px += 1
    py += 1
    num_ghosts = 2 * (px + 1) + 2 * (py + 1)

    total_edges = px * (py + 1) + py * (px + 1)
    total_nodes = px * py + 2 * px + 2 * py
    internal_nodes = px * py
    E10 = np.zeros((total_edges, total_nodes))

    phi_left = range(internal_nodes, internal_nodes + py)
    phi_right = range(internal_nodes + py, internal_nodes + 2 * py)
    phi_bottom = range(internal_nodes + 2 * py, internal_nodes + 2 * py + px)
    phi_top = range(internal_nodes + 2 * py + px, internal_nodes + 2 * py + 2 * px)

    for i in range(px):
        for j in range(py + 1):
            edgeij = (px + 1) * py + i * (py + 1) + j
            if j == 0:
                E10[edgeij, phi_bottom[i]] = 1
                E10[edgeij, i * py + j] = -1
            elif j == py:
                E10[edgeij, i * py + j - 1] = 1
                E10[edgeij, phi_top[i]] = -1
            else:
                E10[edgeij, i * py + j - 1] = 1
                E10[edgeij, i * py + j] = -1

    for i in range(px + 1):
        for j in range(py):
            edgeij = i * py + j
            if i == 0:
                E10[edgeij, phi_left[j]] = 1
                E10[edgeij, i * py + j] = -1
            elif i == px:
                E10[edgeij, (i - 1) * py + j] = 1
                E10[edgeij, phi_right[j]] = -1
            else:
                E10[edgeij, (i - 1) * py + j] = 1
                E10[edgeij, i * py + j] = -1
    E_ghosts = np.zeros((num_ghosts, total_nodes))
    E_10_with_ghosts = np.vstack((E10, E_ghosts))
    return -E_10_with_ghosts
egE10 = d_10_ext_gauss_inner((px, py))
egE10_assembled = assemble(egE10, (func_space_eg1, func_space_eg0))

p1  = Form(func_space_eg1)
p1.cochain = egE10_assembled.dot(f0.cochain)
#p1.reconstruct(xi, eta)
#(x, y), data_dx, data_dy = p1.export_to_plot()
#plt.contourf(x, y, data_dx)
#plt.title('4.1). ext_gauss \\tilde{p}^{(1)} dx')
#plt.colorbar()
#plt.show()
#
#plt.contourf(x, y, data_dy)
#plt.title('4.2). ext_gauss \\tilde{p}^{(1)} dy')
#plt.colorbar()
#plt.show()

# %% 
H11eg = -hodge(func_space_eg1)
H11eg_assembled   = assemble(H11eg  , (func_space_gl1, func_space_eg1))
pn1 = Form(func_space_gl1)
pn1.cochain = H11eg_assembled.dot(p1.cochain[:p1.function_space.num_internal_dof])
#
#pn1.reconstruct(xi, eta)
#(x, y), data_dx, data_dy = pn1.export_to_plot()
#plt.contourf(x, y, data_dx)
#plt.title('5.1). lobatto p^{(n-1)} dx')
#plt.colorbar()
#plt.show()
#
#plt.contourf(x, y, data_dy)
#plt.title('5.2). lobatto p^{(n-1)} dy')
#plt.colorbar()
#plt.show()


# %% ui1
H11gl = -hodge(func_space_gl1)
H11gl_assembled   = assemble(H11gl  , (func_space_eg1, func_space_gl1))

u1 = Form(func_space_eg1)
u1_cochian_internal = H11gl_assembled.dot(un1.cochain)
u1.cochain = np.concatenate((u1_cochian_internal, np.zeros(
        u1.function_space.num_dof - u1.basis.num_basis * u1.mesh.num_elements)), axis=0)
u1.reconstruct(xi, eta)
(x, y), data_dx, data_dy = u1.export_to_plot()
plt.contourf(x, y, data_dx)
plt.title('6.1). ext_gauss \\tilde{u}^{(1)} dx')
plt.colorbar()
plt.show()

plt.contourf(x, y, data_dy)
plt.title('6.2). ext_gauss \\tilde{u}^{(1)} dy')
plt.colorbar()
plt.show()
def UBC(mesh, s, p, position):
    def pullbackedfun_dx(xi, eta):
        x, y = mesh.mapping(xi, eta, s)
        return -u(x, y)

    def pullbackedfun_dy(xi, eta):
        x, y = mesh.mapping(xi, eta, s)
        return v(x, y)

    def fun2bint_dxi(xi, eta):
        return pullbackedfun_dx(xi, eta) * mesh.dx_dxi(xi, eta, s)  + pullbackedfun_dy(xi, eta) * mesh.dy_dxi(xi, eta, s)

    def fun2bint_deta(xi, eta):
        return pullbackedfun_dx(xi, eta) * mesh.dx_deta(xi, eta, s) + pullbackedfun_dy(xi, eta) * mesh.dy_deta(xi, eta, s)
    
    UBC_s = np.zeros(shape = (p+1))
    extended_gauss_nodes, _ = extended_gauss_quad(p-1)
    if position == 'Left':
        for i in range(p+1):
#            print("hello, Left world")
            def fun2bint_deta_BC(eta):
                return fun2bint_deta(-1, eta)
            UBC_s[i] = quad(fun2bint_deta_BC, extended_gauss_nodes[i], extended_gauss_nodes[i+1] )[0]
    elif position == 'Right':
        for i in range(p+1):
#            print("hello, Right world")
            def fun2bint_deta_BC(eta):
                return fun2bint_deta(+1, eta)
            UBC_s[i] = quad(fun2bint_deta_BC, extended_gauss_nodes[i], extended_gauss_nodes[i+1] )[0]
    elif position == 'Bottom':
        for i in range(p+1):
#            print("hello, Bottom world")
            def fun2bint_dxi_BC(xi):
                return fun2bint_dxi(xi, -1)
            UBC_s[i] = quad(fun2bint_dxi_BC, extended_gauss_nodes[i], extended_gauss_nodes[i+1] )[0]
    elif position == 'Top':
        for i in range(p+1):
#            print("hello, Top world")
            def fun2bint_dxi_BC(xi):
                return fun2bint_dxi(xi, +1)
            UBC_s[i] = quad(fun2bint_dxi_BC, extended_gauss_nodes[i], extended_gauss_nodes[i+1] )[0]
    return UBC_s

def extended_gauss1_general_boundary_edges(mesh, p, gathering_matrix):
    p+=1
    nx = mesh.n_x
    ny = mesh.n_y

    Left   = np.zeros( shape = (ny*(p+1)), dtype=np.int32 )
    Right  = np.zeros( shape = (ny*(p+1)), dtype=np.int32 )
    Bottom = np.zeros( shape = (nx*(p+1)), dtype=np.int32 )
    Top    = np.zeros( shape = (nx*(p+1)), dtype=np.int32 )
    
    M = 2 * p * (p+1)
    N = p+1
    
    UBC_L = UBC_R = UBC_B = UBC_T = 0
    
    for J in range(ny):
        eleidLeft  = J
        Left[  J*N : J*N + N ] = gathering_matrix[ eleidLeft , M +2*N : M +3*N ]
        UBC_s=UBC(mesh, eleidLeft, p, "Left")
        if UBC_L is 0:
            UBC_L= UBC_s
        else:
            UBC_L = np.hstack((UBC_L, UBC_s))
        
        eleidRight = (nx-1)*ny + J 
        Right[ J*N : J*N + N ] = gathering_matrix[ eleidRight, M +3*N : M +4*N ]
        UBC_s=UBC(mesh, eleidRight, p, "Right")
        if UBC_R is 0:
            UBC_R= UBC_s
        else:
            UBC_R = np.hstack((UBC_R, UBC_s))

    for I in range(nx):
        eleidBottom = I*ny
        Bottom[ I*N : I*N + N ] = gathering_matrix[ eleidBottom, M     : M +   N ]
        UBC_s=UBC(mesh, eleidBottom, p, "Bottom")
        if UBC_B is 0:
            UBC_B= UBC_s
        else:
            UBC_B = np.hstack((UBC_B, UBC_s))
        
        eleidTop = I*ny + ny -1
        Top[ I*N : I*N + N ]    = gathering_matrix[ eleidTop   , M + N : M + 2*N ] 
        UBC_s=UBC(mesh, eleidTop, p, "Top")
        if UBC_T is 0:
            UBC_T= UBC_s
        else:
            UBC_T = np.hstack((UBC_T, UBC_s))

    return np.vstack((Left, UBC_L)), np.vstack((Right, UBC_R)), np.vstack((Bottom, UBC_B)), np.vstack((Top, UBC_T))

Left, Right, Bottom, Top = extended_gauss1_general_boundary_edges(mesh, px, u1.function_space.dof_map.dof_map)
Boundaryedgs = np.hstack( (Left, Right, Bottom, Top) )
for i in range(np.shape(Boundaryedgs)[1]):
    u1.cochain[int(Boundaryedgs[0, i])] =Boundaryedgs[1, i]

# %% w2
def d_21_ext_gauss_inner(p):
    px, py = p
    px += 1
    py += 1
    internal_edges = (px + 1) * py + (py + 1) * px
    ghost_edges = (px + 1 + py + 1) * 2
    total_edges = internal_edges + ghost_edges
    total_volumes = (px + 1) * (py + 1)

    E21 = np.zeros((total_volumes, total_edges))

    edge_bottom = range(internal_edges, internal_edges + px + 1)
    edge_top = range(internal_edges + px + 1, internal_edges + 2 * px + 2 * 1)
    edge_left = range(internal_edges + 2 * px + 2 * 1, internal_edges + 2 * px + 2 * 1 + py + 1)
    edge_right = range(internal_edges + 2 * px + 2 * 1 + py + 1,
                       internal_edges + 2 * px + 2 * py + 4 * 1)
    for i in range(px + 1):
        for j in range(py + 1):
            volij = i * (py + 1) + j

            if i == 0:
                edgeij_left = edge_left[j]
            else:
                edgeij_left = (px + 1) * py + (i - 1) * (py + 1) + j

            if i == px:
                edgeij_right = edge_right[j]
            else:
                edgeij_right = (px + 1) * py + i * (py + 1) + j

            if j == 0:
                edgeij_bottom = edge_bottom[i]
            else:
                edgeij_bottom = i * py + (j - 1)

            if j == py:
                edgeij_top = edge_top[i]
            else:
                edgeij_top = i * py + j

            E21[volij, edgeij_bottom] = +1
            E21[volij, edgeij_top]    = -1
            E21[volij, edgeij_left]   = -1
            E21[volij, edgeij_right]  = +1
    return E21
egE21 = d_21_ext_gauss_inner( (px, py) )
egE21_assembled = assemble(egE21, (func_space_eg2, func_space_eg1))

w2 = Form(func_space_eg2)
w2.cochain = egE21_assembled.dot(u1.cochain)
w2.reconstruct(xi, eta)
(x, y), data = w2.export_to_plot()
plt.contourf(x, y, data)
plt.title('7). ext_gauss \\tilde{w}^{(2)} dx \wedge dy')
plt.colorbar()
plt.show()

# %% wn2
H02eg = hodge(func_space_eg2)
H02eg_assembled   = assemble(H02eg  , (func_space_gl0, func_space_eg2))


wn2 = Form(func_space_gl0)
wn2.cochain = H02eg_assembled.dot(w2.cochain)
#wn2.reconstruct(xi, eta)
#(x, y), data = wn2.export_to_plot()
#plt.contourf(x, y, data)
#plt.title('8). lobatto w^{(n-2)}')
#plt.colorbar()
#plt.show()

# %%
def d_10_lobatto_outer(p):
    px, py = p
    total_edges = px * (py + 1) + py * (px + 1)
    total_nodes = (px + 1) * (py + 1)
    E10 = np.zeros((total_edges, total_nodes))

    for i in range(px):
        for j in range(py + 1):
            edgeij = i * (py + 1) + j
            node_1 = i * (py + 1) + j
            node_2 = (i + 1) * (py + 1) + j

            E10[edgeij, node_1] = +1
            E10[edgeij, node_2] = -1

    for i in range(px + 1):
        for j in range(py):
            edgeij = px * (py + 1) + py * i + j
            node_1 = i * (py + 1) + j
            node_2 = i * (py + 1) + j + 1

            E10[edgeij, node_1] = +1
            E10[edgeij, node_2] = -1

    return -E10
glE10 = d_10_lobatto_outer((px+1, py+1))
glE10_assembled = assemble(glE10, (func_space_gl1, func_space_gl0))

qn1 = Form(func_space_gl1)
qn1.cochain = glE10_assembled.dot(wn2.cochain)
#qn1.reconstruct(xi, eta)
#(x, y), data_dx, data_dy = qn1.export_to_plot()
#plt.contourf(x, y, data_dx)
#plt.title('9.1). lobatto q^{(n-1)} dx')
#plt.colorbar()
#plt.show()
#
#plt.contourf(x, y, data_dy)
#plt.title('9.2). lobatto q^{(n-1)} dy')
#plt.colorbar()
#plt.show()
#print('max:',np.max(data_dy))
#print('min:',np.min(data_dy))

## %% 
##
##q1 = Form(func_space_eg1)
##q1_cochian_internal = H11gl_assembled.dot(qn1.cochain)
##q1.cochain = np.concatenate((q1_cochian_internal, np.zeros(
##        q1.function_space.num_dof - q1.basis.num_basis * q1.mesh.num_elements)), axis=0)
##q1.reconstruct(xi, eta)
##(x, y), data_dx, data_dy = q1.export_to_plot()
##plt.contourf(x, y, data_dx)
##plt.title('9.1). ext_gauss \\tilde{q}^{(1)} dx')
##plt.colorbar()
##plt.show()
##
##plt.contourf(x, y, data_dy)
##plt.title('9.2). ext_gauss \\tilde{q}^{(1)} dy')
##plt.colorbar()
##plt.show()
##
##
# %% sol
#r1 = Form(func_space_gl1)
#r1.cochain = pn1.cochain + qn1.cochain
#r1.reconstruct(xi, eta)
#(x, y), data_dx, data_dy = r1.export_to_plot()
#plt.contourf(x, y, data_dx)
#plt.title('10.1). ext_gauss \\tilde{r}^{(1)} dx')
#plt.colorbar()
#plt.show()
#
#plt.contourf(x, y, data_dy)
#plt.title('10.2). ext_gauss \\tilde{r}^{(1)} dy')
#plt.colorbar()
#plt.show()
#
#print('max:',np.max(data_dy))
#print('min:',np.min(data_dy))

