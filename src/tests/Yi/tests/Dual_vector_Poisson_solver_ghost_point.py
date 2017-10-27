# -*- coding: utf-8 -*-
"""
(SHORT NAME EXPLANATION)
    
>>>DOCTEST COMMANDS 
(THE TEST ANSWER)

@author: Yi Zhang （张仪）. Created on Thu Jul  6 16:00:33 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
from function_space import FunctionSpace
import numpy as np
from mesh import CrazyMesh
from forms import Form
from hodge import hodge
from coboundaries import d
from assemble import assemble
from _assembling import assemble_, integral1d_
import matplotlib.pyplot as plt
from quadrature import extended_gauss_quad
from scipy.integrate import quad
from sympy import Matrix
import scipy.io
from scipy import sparse
from inner_product import inner

# %% exact solution define
# u^{(1)} = { u,  v }^T
def u(x,y):
	return   +np.cos(2*np.pi*x) * np.sin(2*np.pi*y)

def v(x,y):
	return   -np.sin(2*np.pi*x) * np.cos(2*np.pi*y)

def r_u(x,y):
    return   -8* np.pi**2 * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)

def r_v(x,y):
    return    8* np.pi**2 * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)

# %% 
def solver(p,n,c):
    print("----------------------------------------------------")
    print('p=',p,';n=',n,';c=',c,':')
    px = py = p
    nx = ny = n
    mesh = CrazyMesh( 2, (nx, ny), ((0, 1), (0, 1)), c )
#    xi = eta = np.linspace( -1, 1, np.ceil(100 / (nx * ny)) + 1 )
    
    # %% define function space
#    func_space_g0  = FunctionSpace(mesh, '0-gauss'  , (px + 1, py + 1), is_inner=False)
#    func_space_g1  = FunctionSpace(mesh, '1-gauss'  , (px + 1, py + 1), is_inner=False)
    func_space_gl0 = FunctionSpace(mesh, '0-lobatto', (px + 1, py + 1), is_inner=False, separated_dof=False)
    func_space_gl1 = FunctionSpace(mesh, '1-lobatto', (px + 1, py + 1), is_inner=False)
    func_space_gl2 = FunctionSpace(mesh, '2-lobatto', (px + 1, py + 1), is_inner=False)
    func_space_eg0 = FunctionSpace(mesh, '0-ext_gauss', (px, py))
#    func_space_eg1 = FunctionSpace(mesh, '1-ext_gauss', (px, py))
#    func_space_eg2 = FunctionSpace(mesh, '2-ext_gauss', (px, py))
    
    # %%
    un1 = Form(func_space_gl1)
    f0  = Form(func_space_eg0)
    fn  = Form(func_space_gl2)
    wn2 = Form(func_space_gl0)
    #wn2 = Form(func_space_g0)
    
    un1_exact = Form(func_space_gl1)
    un1_exact.discretize((v,u))
    rn1 = Form(func_space_gl1)
    rn1.discretize((r_v, r_u))
    
    # %% LHS 24 and LHS 42
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
    
                E21[volij, edge_left]   = -1
                E21[volij, edge_right]  = +1
                E21[volij, edge_bottom] = +1
                E21[volij, edge_top]    = -1
        return E21
    glE21 = d_21_lobatto_outer((px+1, py+1))
    
    W0n = fn.basis.wedged(f0.basis)
    LHS13_local = W0n.dot(glE21)
    LHS31_local = LHS13_local.T
    
    Wb = integral1d_(1, ('lobatto_edge',p+1), ('gauss_node', p+1), ('gauss', p+1))
    LHS31_add   = sparse.lil_matrix( ( 2*(p+1)*(p+2) , 4*(p+1)) )
    M = (p+1)*(p+2)
    N =  p+1
    E =  p+2
    for i in range(N):
        for j in range(N):
            # left
            LHS31_add[ M + i, j]           =  + Wb[i,j]
            # right
            LHS31_add[ -N+i, N + j]        =  - Wb[i,j]
            # bottom
            LHS31_add[ i*E , 2*N + j]      =  - Wb[i,j]
    #        # top
            LHS31_add[ (i+1)*E-1, 3*N + j] =  + Wb[i,j]
            
    LHS31_local = sparse.hstack( (LHS31_local,LHS31_add) )
    LHS31       = assemble( LHS31_local.toarray(), (un1.function_space, f0.function_space))
    
    LHS13_local = sparse.vstack( (LHS13_local,LHS31_add.T) )
    LHS13       = sparse.lil_matrix( assemble( LHS13_local.toarray(), (f0.function_space, un1.function_space)))
    
    # %% LHS 11
    f0.basis.quad_grid = ('gauss',px+1)
    egM0 = inner(f0.basis, f0.basis)
    LHS11       = assemble(egM0, (func_space_eg0, func_space_eg0))
    temp = sparse.lil_matrix((f0.function_space.num_dof,f0.function_space.num_dof))
    temp[0:f0.function_space.num_internal_dof,0:f0.function_space.num_internal_dof]= LHS11
    LHS11 = temp
    
    # %% LHS 22
    wn2.basis.quad_grid = ('lobatto',px+1)
    glMn2 = inner(wn2.basis, wn2.basis)
    LHS22 = assemble(glMn2, (func_space_gl0, func_space_gl0))
    
    #wn2.basis.quad_grid = ( 'gauss', px+1 )
    #glMn2 = inner(wn2.basis, wn2.basis)
    #LHS22 = assemble(glMn2, (func_space_g0, func_space_g0))
    
    # %% LHS 23 , 32
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
    glE10 = assemble(glE10, (func_space_gl1, func_space_gl0))
    
    glMn1 = inner(un1.basis, un1.basis)
    glMn1 = assemble(glMn1, (func_space_gl1, func_space_gl1))
    LHS32 = glMn1.dot(glE10)
    LHS23 = LHS32.T
    
    #def d_10_gauss_outer(p):
    #    px, py = p
    #    total_edges = px * (py + 1) + py * (px + 1)
    #    total_nodes = (px + 1) * (py + 1)
    #    E10 = np.zeros((total_edges, total_nodes))
    #
    #    for i in range(px):
    #        for j in range(py + 1):
    #            edgeij = i * (py + 1) + j
    #            node_1 = i * (py + 1) + j
    #            node_2 = (i + 1) * (py + 1) + j
    #
    #            E10[edgeij, node_1] = +1
    #            E10[edgeij, node_2] = -1
    #
    #    for i in range(px + 1):
    #        for j in range(py):
    #            edgeij = px * (py + 1) + py * i + j
    #            node_1 = i * (py + 1) + j
    #            node_2 = i * (py + 1) + j + 1
    #
    #            E10[edgeij, node_1] = +1
    #            E10[edgeij, node_2] = -1
    #
    #    return -E10
    #
    #glE10 = d_10_gauss_outer((px+1, py+1))
    #glE10 = assemble(glE10, (func_space_g1, func_space_g0))
    #
    #glMn1 = inner(un1.basis, Form(func_space_g1).basis )
    #glMn1 = assemble(glMn1, (func_space_gl1, func_space_g1))
    #LHS32 = glMn1.dot(glE10)
    #LHS23 = LHS32.T
    
    # %% boundary edges
    def lobatto_boundary_edges(mesh, p, gathering_matrix):
        
        nx = mesh.n_x
        ny = mesh.n_y
    
        Left   = np.zeros( shape = (ny*(p)), dtype=np.int32 )
        Right  = np.zeros( shape = (ny*(p)), dtype=np.int32 )
        Bottom = np.zeros( shape = (nx*(p)), dtype=np.int32 )
        Top    = np.zeros( shape = (nx*(p)), dtype=np.int32 )
        
        M = p * (p+1)
        N = p
    
        for J in range(ny):
            eleidLeft  = J
            Left[  J*N : J*N + N ] = gathering_matrix[ eleidLeft , M : M + N ]
            
            eleidRight = (nx-1)*ny + J 
            Right[ J*N : J*N + N ] = gathering_matrix[ eleidRight, -N : ]
    
        for I in range(nx):
            eleidBottom = I*ny
            Bottom[ I*N : I*N + N ] = gathering_matrix[ eleidBottom, 0 : M: N+1 ]
            
            eleidTop = I*ny + ny -1
            Top[ I*N : I*N + N ]    = gathering_matrix[ eleidTop   , N : M: N+1 ] 
    
        return Left, Right, Bottom, Top
    
    eLeft, eRight, eBottom, eTop = lobatto_boundary_edges(mesh, p+1, un1.function_space.dof_map.dof_map)
    
    
    def ext_gauss_boundary_points(mesh, p, gathering_matrix):
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
    nLeft, nRight, nBottom, nTop = ext_gauss_boundary_points(mesh, p+1, f0.function_space.dof_map.dof_map)
    
    RHS1 = np.zeros(shape=(f0.function_space.num_dof,1))
    for i in range(np.size(nLeft)):
        LHS13[nLeft[i],:] = 0
        LHS13[nLeft[i],eLeft[i]] = 1
        RHS1 [nLeft[i]] = un1_exact.cochain[eLeft[i]]
        
        LHS13[nRight[i],:] = 0
        LHS13[nRight[i],eRight[i]] = 1
        RHS1 [nRight[i]] = un1_exact.cochain[eRight[i]]
        
        LHS13[nBottom[i],:] = 0
        LHS13[nBottom[i],eBottom[i]] = 1
        RHS1 [nBottom[i]] = un1_exact.cochain[eBottom[i]]
        
        LHS13[nTop[i],:] = 0
        LHS13[nTop[i],eTop[i]] = 1
        RHS1 [nTop[i]] = un1_exact.cochain[eTop[i]]
        
    # %%
    LHS = sparse.bmat(
            [[LHS11,
              sparse.csc_matrix((f0.function_space.num_dof,wn2.function_space.num_dof)),
              LHS13],
    
            [ sparse.csc_matrix((wn2.function_space.num_dof,f0.function_space.num_dof)),
              LHS22,
             -LHS23],
            
            [ LHS31,
             -LHS32,
              sparse.csc_matrix((un1.function_space.num_dof,un1.function_space.num_dof))]])
    
    # %% RHS2
    RHS2 = np.zeros(shape=(wn2.function_space.num_dof,1))
    
    # %% RHS3
    RHS3 = glMn1.dot(rn1.cochain)
    RHS3 = np.expand_dims(RHS3, axis=1)
    
    #%% solve it
    RHS =     np.vstack((RHS1, RHS2, RHS3))
    print("LHS shape:", np.shape(LHS))
    #    
    LHS = sparse.csr_matrix(LHS)
    print("------ solve the square sparse system:......")
    Res = sparse.linalg.spsolve(LHS,RHS)
    
    # %% split the Res
    f0.cochain = Res[:f0.function_space.num_dof ].reshape(f0.function_space.num_dof)
    wn2.cochain = Res[f0.function_space.num_dof:-un1.function_space.num_dof ].reshape(wn2.function_space.num_dof)
    un1.cochain = Res[-un1.function_space.num_dof:].reshape(un1.function_space.num_dof)
    
    # %%
    #f0.reconstruct(xi, eta)
    #(x, y), data = f0.export_to_plot()
    #plt.contourf(x, y, data)
    #plt.title("3). ext_gauss \\tilde{f}^{(0)}")
    #plt.colorbar()
    #plt.show()
    #
    #wn2.reconstruct(xi, eta)
    #(x, y), data = wn2.export_to_plot()
    #plt.contourf(x, y, data)
    #plt.title('8). lobatto w^{(n-2)}')
    #plt.colorbar()
    #plt.show()
    #
    #un1.reconstruct(xi, eta)
    #(x, y), data_dx, data_dy = un1.export_to_plot()
    #plt.contourf(x, y, data_dx)
    #plt.title('1.1). lobatto u^{(n-1)} dx')
    #plt.colorbar()
    #plt.show()
    #
    #plt.contourf(x, y, data_dy)
    #plt.title('1.2). lobatto u^{(n-1)} dy')
    #plt.colorbar()
    #plt.show()
    
    # %%
    L2_error_un1 = un1.l_2_norm((v, u), ('lobatto', p + 5))[0]
    
    def wn2_fun(x,y): return 0*x*y
    L2_error_wn2 = wn2.l_2_norm(wn2_fun, ('lobatto', p + 5))[0]
    
    def f0_fun(x,y): return 4 * np.pi * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    L2_error_f0 = f0.l_2_norm(f0_fun, ('gauss', p + 5))[0]
    
    print("------ L2_error_un1 =", L2_error_un1)
    print("------ L2_error_wn2 =", L2_error_wn2)
    print("------ L2_error_f0  =", L2_error_f0)
    return L2_error_un1, L2_error_wn2, L2_error_f0

if __name__ == "__main__":
    i = 0
    for p in [2]:
        for c in [0, 0.3]:
            for n in [2,3,5,7,10,15,22,32,46]:
                temp_L2_error_un1, temp_L2_error_wn2, temp_L2_error_f0 = solver(p,n,c)
                if i == 0:
                    p_c_n_l2un1_l2wn2_l2f0 = np.array( [p,c,n,temp_L2_error_un1,temp_L2_error_wn2, temp_L2_error_f0] )
                else:
                    p_c_n_l2un1_l2wn2_l2f0 = np.vstack((p_c_n_l2un1_l2wn2_l2f0, np.array([p, c, n, temp_L2_error_un1, temp_L2_error_wn2, temp_L2_error_f0])))
                i += 1 

    scipy.io.savemat('vector_Poisson_p_convergence_07011_p2.mat', mdict={'p_c_n_l2un1_l2wn2_l2f0': p_c_n_l2un1_l2wn2_l2f0})