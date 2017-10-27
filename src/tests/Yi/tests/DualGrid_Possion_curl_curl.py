import path_magic
from function_space import FunctionSpace
import numpy as np
from mesh import CrazyMesh
from forms import Form
from hodge import hodge
from coboundaries import d
from assemble import assemble
import matplotlib.pyplot as plt
from quadrature import extended_gauss_quad
from scipy.integrate import quad
from sympy import Matrix
import scipy.io
from scipy import sparse
import scipy as sp
# %%
def pfun(x, y):
    return -np.sin(np.pi * x) * np.sin(np.pi * y)

def ui_dx(x,y):
	return np.pi * np.sin(np.pi*x) * np.cos(np.pi*y)

def ui_dy(x,y):
	return -np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)

def ffun(x, y):
    return 2 * np.pi**2 * np.sin( np.pi * x) * np.sin( np.pi * y)

# %% THis is the main propgram body
def solver(p,n,c):
    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("Start curl curl solver @ p=", p)
    print("                       @ n=", n)
    print("                       @ c=", c)
    px = py = p
    nx = n
    ny = n
    mesh = CrazyMesh(2, (nx, ny), ((-1, 1), (-1, 1)), c)
    xi = eta = np.linspace(-1, 1, np.ceil(500 / (nx * ny)) + 1)
    # %% exact p(0)
    func_space_gl0 = FunctionSpace(mesh, '0-lobatto', (px + 1, py + 1), is_inner=False)
    p0_exact = Form(func_space_gl0)
    p0_exact.discretize(pfun)
    p0_exact.reconstruct(xi, eta)
    (x, y), data = p0_exact.export_to_plot()
    plt.contourf(x, y, data)
    plt.title('exact lobatto 0-form, p0')
    plt.colorbar()
    plt.show()
    
    # %% p(0)
    func_space_gl0 = FunctionSpace(mesh, '0-lobatto', (px + 1, py + 1), is_inner=False)
    p0 = Form(func_space_gl0)
    
    # %%
    func_space_gl1 = FunctionSpace(mesh, '1-lobatto', (px + 1, py + 1), is_inner=False)
    uo = Form(func_space_gl1)
    
    # %%
    func_space_eg1 = FunctionSpace(mesh, '1-ext_gauss', (px, py))
    ui = Form(func_space_eg1)
    
    # %%
    func_space_eg2 = FunctionSpace(mesh, '2-ext_gauss', (px, py))
    f2 = Form(func_space_eg2)
    f2_exact = Form(func_space_eg2)
    f2_exact.discretize(ffun)
#    f2_exact.reconstruct(xi, eta)
#    (x, y), data = f2_exact.export_to_plot()
#    plt.contourf(x, y, data)
#    plt.title('exact extended-gauss 2-form, f2')
#    plt.colorbar()
#    plt.show()
    
    # %%
    E10 = d(func_space_gl0)
#    E10_assembled = assemble(mesh, E10, uo.function_space.dof_map.dof_map,p0.function_space.dof_map.dof_map, mode='replace')
    E10_assembled = assemble(E10, (uo.function_space,p0.function_space))
    
    H = hodge(func_space_gl1)
#    H_assembled   = assemble(mesh, H  , ui.function_space.dof_map.dof_map_internal, uo.function_space.dof_map.dof_map)
    H_assembled   = assemble(H  , (ui.function_space, uo.function_space))
    #H_assembled   = np.linalg.inv(H_assembled)
    
    E21 = d(func_space_eg1)
#    E21_assembled = assemble(mesh, E21, f2.function_space.dof_map.dof_map, ui.function_space.dof_map.dof_map, mode = 'replace')
    E21_assembled = assemble(E21, (f2.function_space, ui.function_space))
# %%
#    uo.cochain = E10_assembled.dot(p0_exact.cochain)
#    uo.reconstruct(xi, eta)
#    (x, y), data_dx, data_dy = uo.export_to_plot()
#    plt.contourf(x, y, data_dx)
#    plt.title('exact lobatto 1-form dx')
#    plt.colorbar()
#    plt.show()
#    print('uo_dx max:', np.max(data_dx))
#    print('uo_dx min:', np.min(data_dx))
#    
#    plt.contourf(x, y, data_dy)
#    plt.title('exact lobatto 1-form dy')
#    plt.colorbar()
#    plt.show()
#    print('uo_dy max:', np.max(data_dy))
#    print('uo_dy min:', np.min(data_dy))
#    
#    ui_internal_cochain = H_assembled.dot(uo.cochain)
#    ui.cochain = np.concatenate((ui_internal_cochain, np.zeros(
#        ui.function_space.num_dof - ui.basis.num_basis * ui.mesh.num_elements)), axis=0)
#    ui.reconstruct(xi, eta)
#    (x, y), data_dx, data_dy = ui.export_to_plot()
#    plt.contourf(x, y, data_dx)
#    plt.title('ext_gauss 1-form dx')
#    plt.colorbar()
#    plt.show()
#    print('ui_dx max:', np.max(data_dx))
#    print('ui_dy min:', np.min(data_dx))
#    
#    plt.contourf(x, y, data_dy)
#    plt.title('ext_gauss 1-form dy')
#    plt.colorbar()
#    plt.show()
#    print('ui_dy max:', np.max(data_dy))
#    print('ui_dy min:', np.min(data_dy))
#    def UBC(mesh, s, p, position):
#        def pullbackedfun_dx(xi, eta):
#            x, y = mesh.mapping(xi, eta, s)
#            return ui_dx(x, y)
#    
#        def pullbackedfun_dy(xi, eta):
#            x, y = mesh.mapping(xi, eta, s)
#            return ui_dy(x, y)
#    
#        def fun2bint_dxi(xi, eta):
#            return pullbackedfun_dx(xi, eta) * mesh.dx_dxi(xi, eta, s)  + pullbackedfun_dy(xi, eta) * mesh.dy_dxi(xi, eta, s)
#    
#        def fun2bint_deta(xi, eta):
#            return pullbackedfun_dx(xi, eta) * mesh.dx_deta(xi, eta, s) + pullbackedfun_dy(xi, eta) * mesh.dy_deta(xi, eta, s)
#        
#        UBC_s = np.zeros(shape = (p+1))
#        extended_gauss_nodes, _ = extended_gauss_quad(p-1)
#        if position == 'Left':
#            for i in range(p+1):
#    #            print("hello, Left world")
#                def fun2bint_deta_BC(eta):
#                    return fun2bint_deta(-1, eta)
#                UBC_s[i] = quad(fun2bint_deta_BC, extended_gauss_nodes[i], extended_gauss_nodes[i+1] )[0]
#        elif position == 'Right':
#            for i in range(p+1):
#    #            print("hello, Right world")
#                def fun2bint_deta_BC(eta):
#                    return fun2bint_deta(+1, eta)
#                UBC_s[i] = quad(fun2bint_deta_BC, extended_gauss_nodes[i], extended_gauss_nodes[i+1] )[0]
#        elif position == 'Bottom':
#            for i in range(p+1):
#    #            print("hello, Bottom world")
#                def fun2bint_dxi_BC(xi):
#                    return fun2bint_dxi(xi, -1)
#                UBC_s[i] = quad(fun2bint_dxi_BC, extended_gauss_nodes[i], extended_gauss_nodes[i+1] )[0]
#        elif position == 'Top':
#            for i in range(p+1):
#    #            print("hello, Top world")
#                def fun2bint_dxi_BC(xi):
#                    return fun2bint_dxi(xi, +1)
#                UBC_s[i] = quad(fun2bint_dxi_BC, extended_gauss_nodes[i], extended_gauss_nodes[i+1] )[0]
#        return UBC_s
#    
#    def extended_gauss1_general_boundary_edges(mesh, p, gathering_matrix):
#        p+=1
#        nx = mesh.n_x
#        ny = mesh.n_y
#    
#        Left   = np.zeros( shape = (ny*(p+1)), dtype=np.int32 )
#        Right  = np.zeros( shape = (ny*(p+1)), dtype=np.int32 )
#        Bottom = np.zeros( shape = (nx*(p+1)), dtype=np.int32 )
#        Top    = np.zeros( shape = (nx*(p+1)), dtype=np.int32 )
#        
#        M = 2 * p * (p+1)
#        N = p+1
#        
#        UBC_L = UBC_R = UBC_B = UBC_T = 0
#        
#        for J in range(ny):
#            eleidLeft  = J
#            Left[  J*N : J*N + N ] = gathering_matrix[ eleidLeft , M +2*N : M +3*N ]
#            UBC_s=UBC(mesh, eleidLeft, p, "Left")
#            if UBC_L is 0:
#                UBC_L= UBC_s
#            else:
#                UBC_L = np.hstack((UBC_L, UBC_s))
#            
#            eleidRight = (nx-1)*ny + J 
#            Right[ J*N : J*N + N ] = gathering_matrix[ eleidRight, M +3*N : M +4*N ]
#            UBC_s=UBC(mesh, eleidRight, p, "Right")
#            if UBC_R is 0:
#                UBC_R= UBC_s
#            else:
#                UBC_R = np.hstack((UBC_R, UBC_s))
#    
#        for I in range(nx):
#            eleidBottom = I*ny
#            Bottom[ I*N : I*N + N ] = gathering_matrix[ eleidBottom, M     : M +   N ]
#            UBC_s=UBC(mesh, eleidBottom, p, "Bottom")
#            if UBC_B is 0:
#                UBC_B= UBC_s
#            else:
#                UBC_B = np.hstack((UBC_B, UBC_s))
#            
#            eleidTop = I*ny + ny -1
#            Top[ I*N : I*N + N ]    = gathering_matrix[ eleidTop   , M + N : M + 2*N ] 
#            UBC_s=UBC(mesh, eleidTop, p, "Top")
#            if UBC_T is 0:
#                UBC_T= UBC_s
#            else:
#                UBC_T = np.hstack((UBC_T, UBC_s))
#    
#        return np.vstack((Left, UBC_L)), np.vstack((Right, UBC_R)), np.vstack((Bottom, UBC_B)), np.vstack((Top, UBC_T))
#    
#    Left, Right, Bottom, Top = extended_gauss1_general_boundary_edges(mesh, px, ui.function_space.dof_map.dof_map)
#    Boundaryedgs = np.hstack( (Left, Right, Bottom, Top) )
#    for i in range(np.shape(Boundaryedgs)[1]):
#        ui.cochain[int(Boundaryedgs[0, i])] =Boundaryedgs[1, i]
#    
#    f2.cochain = E21_assembled.dot(ui.cochain)
#    f2.reconstruct(xi, eta)
#    (x, y), data = f2.export_to_plot()
#    plt.contourf(x, y, data)
#    plt.title('exact extended-gauss 2-form, f2')
#    plt.colorbar()
#    plt.show()
    # %%
    # system:
    #  |  I     -H       0  |   | ui |      | 0 |
    #  |                    |   |    |      |   |
    #  |  0      I     -E10 | * | uo |   =  | 0 |
    #  |                    |   |    |      |   |
    #  | E21     0       0  |   | p  |      | f |
    ui_num_dof_internal = ui.basis.num_basis * ui.mesh.num_elements
    ui_num_dof_external = ui.function_space.num_dof - ui_num_dof_internal

#    LHS1 = np.hstack((   np.eye(ui_num_dof_internal) , 
#                         np.zeros((ui_num_dof_internal, ui_num_dof_external)),
#                        -H_assembled ,
#                         np.zeros((ui_num_dof_internal, p0.function_space.num_dof))   ))
    LHS1 = sparse.hstack((   sparse.eye(ui_num_dof_internal) , 
                             sparse.csc_matrix((ui_num_dof_internal, ui_num_dof_external)),
                            -H_assembled ,
                             sparse.csc_matrix((ui_num_dof_internal, p0.function_space.num_dof))   ))
    
#    LHS2 = np.hstack((  np.zeros((uo.function_space.num_dof, ui.function_space.num_dof )) , 
#                        np.eye(uo.function_space.num_dof) , 
#                       -E10_assembled    ))
    LHS2 = sparse.hstack((  sparse.csc_matrix((uo.function_space.num_dof, ui.function_space.num_dof )) , 
                            sparse.eye(uo.function_space.num_dof) , 
                           -E10_assembled    ))
    
    
#    LHS3 = np.hstack((  E21_assembled , 
#                        np.zeros((f2.function_space.num_dof, uo.function_space.num_dof )) , 
#                        np.zeros((f2.function_space.num_dof, f2.function_space.num_dof ))    ))
    LHS3 = sparse.hstack((  E21_assembled , 
                            sparse.csc_matrix((f2.function_space.num_dof, uo.function_space.num_dof )) , 
                            sparse.csc_matrix((f2.function_space.num_dof, f2.function_space.num_dof ))    ))
    
    RHS1 = np.zeros((ui_num_dof_internal, 1 ))
    RHS2 = np.zeros((uo.function_space.num_dof, 1 ))
    RHS3 = f2_exact.cochain.reshape( (f2_exact.function_space.num_dof,1) )
    
 # %% boundary edges
#    def UBC(mesh, s, p, position):
#        def pullbackedfun_dx(xi, eta):
#            x, y = mesh.mapping(xi, eta, s)
#            return ufun_u(x, y)
#    
#        def pullbackedfun_dy(xi, eta):
#            x, y = mesh.mapping(xi, eta, s)
#            return ufun_v(x, y)
#    
#        def fun2bint_dxi(xi, eta):
#            return pullbackedfun_dx(xi, eta) * mesh.dx_dxi(xi, eta, s)  + pullbackedfun_dy(xi, eta) * mesh.dy_dxi(xi, eta, s)
#    
#        def fun2bint_deta(xi, eta):
#            return pullbackedfun_dx(xi, eta) * mesh.dx_deta(xi, eta, s) + pullbackedfun_dy(xi, eta) * mesh.dy_deta(xi, eta, s)
#        
#        UBC_s = np.zeros(shape = (p+1))
#        extended_gauss_nodes, _ = extended_gauss_quad(p-1)
#        if position == 'Left':
#            for i in range(p+1):
#    #            print("hello, Left world")
#                def fun2bint_deta_BC(eta):
#                    return fun2bint_deta(-1, eta)
#                UBC_s[i] = quad(fun2bint_deta_BC, extended_gauss_nodes[i], extended_gauss_nodes[i+1] )[0]
#        elif position == 'Right':
#            for i in range(p+1):
#    #            print("hello, Right world")
#                def fun2bint_deta_BC(eta):
#                    return fun2bint_deta(+1, eta)
#                UBC_s[i] = quad(fun2bint_deta_BC, extended_gauss_nodes[i], extended_gauss_nodes[i+1] )[0]
#        elif position == 'Bottom':
#            for i in range(p+1):
#    #            print("hello, Bottom world")
#                def fun2bint_dxi_BC(xi):
#                    return fun2bint_dxi(xi, -1)
#                UBC_s[i] = quad(fun2bint_dxi_BC, extended_gauss_nodes[i], extended_gauss_nodes[i+1] )[0]
#        elif position == 'Top':
#            for i in range(p+1):
#    #            print("hello, Top world")
#                def fun2bint_dxi_BC(xi):
#                    return fun2bint_dxi(xi, +1)
#                UBC_s[i] = quad(fun2bint_dxi_BC, extended_gauss_nodes[i], extended_gauss_nodes[i+1] )[0]
#        return UBC_s
#    
#    def extended_gauss1_general_boundary_edges(mesh, p, gathering_matrix):
#        p+=1
#        nx = mesh.n_x
#        ny = mesh.n_y
#    
#        Left   = np.zeros( shape = (ny*(p+1)), dtype=np.int32 )
#        Right  = np.zeros( shape = (ny*(p+1)), dtype=np.int32 )
#        Bottom = np.zeros( shape = (nx*(p+1)), dtype=np.int32 )
#        Top    = np.zeros( shape = (nx*(p+1)), dtype=np.int32 )
#        
#        M = 2 * p * (p+1)
#        N = p+1
#        
#        UBC_L = UBC_R = UBC_B = UBC_T = 0
#        
#        for J in range(ny):
#            eleidLeft  = J
#            Left[  J*N : J*N + N ] = gathering_matrix[ eleidLeft , M +2*N : M +3*N ]
#            UBC_s=UBC(mesh, eleidLeft, p, "Left")
#            if UBC_L is 0:
#                UBC_L= UBC_s
#            else:
#                UBC_L = np.hstack((UBC_L, UBC_s))
#            
#            eleidRight = (nx-1)*ny + J 
#            Right[ J*N : J*N + N ] = gathering_matrix[ eleidRight, M +3*N : M +4*N ]
#            UBC_s=UBC(mesh, eleidRight, p, "Right")
#            if UBC_R is 0:
#                UBC_R= UBC_s
#            else:
#                UBC_R = np.hstack((UBC_R, UBC_s))
#    
#        for I in range(nx):
#            eleidBottom = I*ny
#            Bottom[ I*N : I*N + N ] = gathering_matrix[ eleidBottom, M     : M +   N ]
#            UBC_s=UBC(mesh, eleidBottom, p, "Bottom")
#            if UBC_B is 0:
#                UBC_B= UBC_s
#            else:
#                UBC_B = np.hstack((UBC_B, UBC_s))
#            
#            eleidTop = I*ny + ny -1
#            Top[ I*N : I*N + N ]    = gathering_matrix[ eleidTop   , M + N : M + 2*N ] 
#            UBC_s=UBC(mesh, eleidTop, p, "Top")
#            if UBC_T is 0:
#                UBC_T= UBC_s
#            else:
#                UBC_T = np.hstack((UBC_T, UBC_s))
#    
#        return np.vstack((Left, UBC_L)), np.vstack((Right, UBC_R)), np.vstack((Bottom, UBC_B)), np.vstack((Top, UBC_T))
#    
#    Left, Right, Bottom, Top = extended_gauss1_general_boundary_edges(mesh, px, ui.function_space.dof_map.dof_map)
#    Boundaryedgs = np.hstack( (Left, Right, Bottom, Top) )
#    
##    LBC = np.zeros( shape = ( np.shape(Boundaryedgs)[1], ui.function_space.num_dof+ uo.function_space.num_dof + p0.function_space.num_dof ) )
#    LBC = sparse.lil_matrix( ( np.shape(Boundaryedgs)[1], ui.function_space.num_dof+ uo.function_space.num_dof + p0.function_space.num_dof ) )
#    
#    RBC = np.zeros( shape = ( np.shape(Boundaryedgs)[1], 1) )
#    for i in range(np.shape(Boundaryedgs)[1]):
#        if i == 0 :
#            LBC[0, ui.function_space.num_dof+ uo.function_space.num_dof] = 1
#            RBC[0] = p0_exact.cochain[0]
#        else:
#            LBC[i, int(Boundaryedgs[0, i])] =1
#            RBC[i] = Boundaryedgs[1, i]
    
# %%
    def PBC(p, nx, ny, gathering_matrix, gathering_matrix_edge):
        p += 1
        
        Left   = np.zeros( shape = (ny*(p+1),4), dtype=np.int32 )
        Right  = np.zeros( shape = (ny*(p+1),4), dtype=np.int32 )
        Bottom = np.zeros( shape = (nx*(p+1),4), dtype=np.int32 )
        Top    = np.zeros( shape = (nx*(p+1),4), dtype=np.int32 )
        
        N = p+1
        P = 2 * p * (p+1)
        Q = (p + 1)
        
        for J in range(ny):
            eleidLeft  = J
            Left[  J*N : J*N + N , 0] = gathering_matrix[ eleidLeft ,  : N ]
            if eleidLeft == 0: # left-bottom corner element
                Left[0, 0] = -1 # left - bottom corner point
                Left[0, 1] = gathering_matrix_edge[0, P + 2*Q]
                Left[0, 2] = gathering_matrix_edge[0, P]
            if eleidLeft == ny-1: # left-top corner element
                Left[-1, 0] = -3 # left _- top corner point
                Left[-1, 1] = gathering_matrix_edge[eleidLeft, P + Q]
                Left[-1, 2] = gathering_matrix_edge[eleidLeft, P + 3*Q -1]
            if eleidLeft >= 0 and eleidLeft < ny - 1:
                Left[J*N + N -1, 0] = -2 # left
                Left[J*N + N -1, 1] = gathering_matrix_edge[eleidLeft, P + 3*Q -1]
                Left[J*N + N -1, 2] = gathering_matrix_edge[eleidLeft, P+Q]
                Left[J*N + N -1, 3] = gathering_matrix_edge[eleidLeft+1, P+2*Q]
            
            eleidRight = (nx-1)*ny + J 
            Right[ J*N : J*N + N , 0] = gathering_matrix[ eleidRight, -N : ]
            if eleidRight == nx*ny - ny: # right bottom element
                Right[0, 0] = -4
                Right[0, 1] = gathering_matrix_edge[eleidRight, P + Q - 1]
                Right[0, 2] = gathering_matrix_edge[eleidRight, P + 3 * Q]
            if eleidRight == nx*ny - 1: # right top element
                Right[-1, 0] = -5
                Right[-1, 1] = gathering_matrix_edge[eleidRight, P + 2*Q -1]
                Right[-1, 2] = gathering_matrix_edge[eleidRight, -1]
            if eleidRight >= nx*ny - ny and eleidRight < nx*ny - 1: # right elements
                Right[J*N + N-1, 0] = -6
                Right[J*N + N-1, 1] = gathering_matrix_edge[eleidRight, -1]
                Right[J*N + N-1, 2] = gathering_matrix_edge[eleidRight, P + 2*Q - 1]
                Right[J*N + N-1, 3] = gathering_matrix_edge[eleidRight+1, P + 3*Q  ]
            
        for I in range(nx):
            eleidBottom = I*ny
            Bottom[ I*N : I*N + N , 0] = gathering_matrix[ eleidBottom, 0 : N**2 : N ]
            if eleidBottom >= 0 and eleidBottom < nx*ny - ny: # bottom elements
                Bottom[I*N + N -1, 0] = -7
                Bottom[I*N + N -1, 1] = gathering_matrix_edge[eleidBottom, P + Q -1]
                Bottom[I*N + N -1, 2] = gathering_matrix_edge[eleidBottom, P + 3*Q]
                Bottom[I*N + N -1, 3] = gathering_matrix_edge[eleidBottom + ny, P]

            eleidTop = I*ny + ny -1
            Top[ I*N : I*N + N, 0 ]    = gathering_matrix[ eleidTop   , N-1 : N**2 : N ] 

            if eleidTop >= 0 and eleidTop < nx*ny-1: # bottom elements
                Top[I*N + N -1, 0] = -8
                Top[I*N + N -1, 1] = gathering_matrix_edge[eleidTop, P + 2*Q -1]
                Top[I*N + N -1, 2] = gathering_matrix_edge[eleidTop, -1]
                Top[I*N + N -1, 3] = gathering_matrix_edge[eleidTop + ny, P+Q]

        return Left, Right, Bottom, Top
    
    Left, Right, Bottom, Top = PBC(p, nx, ny, p0_exact.function_space.dof_map.dof_map, ui.function_space.dof_map.dof_map)
    
    Boundarypoints = np.vstack( (Left, Right, Bottom, Top) )

    LBC = sparse.lil_matrix( ( np.shape(Boundarypoints)[0], ui.function_space.num_dof+ uo.function_space.num_dof + p0.function_space.num_dof ) )
    RBC = np.zeros( shape = ( np.shape(Boundarypoints)[0], 1) )
    for i in range(np.shape(Boundarypoints)[0]):
        if Boundarypoints[i, 0] == -1:
            LBC[i, Boundarypoints[i, 1]] = +1
            LBC[i, Boundarypoints[i, 2]] = +1
            RBC[i] = 0
        elif Boundarypoints[i,0] == -2:
            LBC[i, Boundarypoints[i, 1]] = -2
            LBC[i, Boundarypoints[i, 2]] = +1
            LBC[i, Boundarypoints[i, 3]] = +1
            RBC[i] = 0
        elif Boundarypoints[i,0] == -3:
            LBC[i, Boundarypoints[i, 1]] = +1
            LBC[i, Boundarypoints[i, 2]] = -1
            RBC[i] = 0
        elif Boundarypoints[i,0] == -4:
            LBC[i, Boundarypoints[i, 1]] = +1
            LBC[i, Boundarypoints[i, 2]] = -1
            RBC[i] = 0
        elif Boundarypoints[i,0] == -5:
            LBC[i, Boundarypoints[i, 1]] = +1
            LBC[i, Boundarypoints[i, 2]] = +1
            RBC[i] = 0
        elif Boundarypoints[i,0] == -6:
            LBC[i, Boundarypoints[i, 1]] = +1
            LBC[i, Boundarypoints[i, 2]] = +1
            LBC[i, Boundarypoints[i, 3]] = -2
            RBC[i] = 0
        elif Boundarypoints[i,0] == -7:
            LBC[i, Boundarypoints[i, 1]] = -2
            LBC[i, Boundarypoints[i, 2]] = +1
            LBC[i, Boundarypoints[i, 3]] = +1
            RBC[i] = 0
        elif Boundarypoints[i,0] == -8:
            LBC[i, Boundarypoints[i, 1]] = +1
            LBC[i, Boundarypoints[i, 2]] = +1
            LBC[i, Boundarypoints[i, 3]] = -2
            RBC[i] = 0
        else:
            LBC[i, ui.function_space.num_dof+ uo.function_space.num_dof +int(Boundarypoints[i,0])] =1
            RBC[i] = p0_exact.cochain[Boundarypoints[i,0]]
            
    # %% 
    def dof_map_crazy_lobatto_point_pair(mesh, p, global_numbering, global_numbering_edges):
        nx, ny = mesh.n_x, mesh.n_y
        
        N = p+1
        
        interface_point_pair = np.zeros( ( ( (nx - 1) * ny + nx * (ny - 1) ) * N, 4 ), dtype=np.int32 )
        
        n = 0
        
        P = 2 * p * (p+1)
        Q = (p + 1)
        
        for i in range(nx - 1):
            for j in range(ny):
                s1 = j + i * ny
                s2 = j + (i + 1) * ny
    #            print(s1, s2)
                for m in range(N):
                    interface_point_pair[n, 0] = global_numbering[s1, N * p + m]
                    interface_point_pair[n, 1] = global_numbering[s2,         m]
                    
                    if j < ny-1 and m == N-1:
                        s3 = s2 + 1
                        interface_point_pair[n, 0] = global_numbering_edges[s1, P + 2 * Q -1 ]
                        interface_point_pair[n, 1] = global_numbering_edges[s1, -1 ]
                        interface_point_pair[n, 2] = global_numbering_edges[s2, P + Q ]
                        interface_point_pair[n, 3] = global_numbering_edges[s3, P + 2* Q ]
                    n += 1
        for i in range(nx):
            for j in range(ny - 1):
                s1 = j + i * ny
                s2 = j + 1 + i * ny
    #            print(s1, s2)
                for m in range(N):
                    interface_point_pair[n, 0] = global_numbering[s1, (m+1)*N -1]
                    interface_point_pair[n, 1] = global_numbering[s2, m*N]
                    n += 1
        return interface_point_pair
    interface_point_pair = dof_map_crazy_lobatto_point_pair(mesh, px+1, p0.function_space.dof_map.dof_map, ui.function_space.dof_map.dof_map)
#    Lintface = np.zeros( shape = ( np.shape( interface_point_pair )[0], ui.function_space.num_dof+ uo.function_space.num_dof + p0.function_space.num_dof ) ) 
    Lintface = sparse.lil_matrix( ( np.shape( interface_point_pair )[0], ui.function_space.num_dof+ uo.function_space.num_dof + p0.function_space.num_dof ) ) 
    #Rintface = np.zeros( shape = ( np.shape( interface_point_pair )[0]-(nx-1)*(ny-1), 1) )
    Rintface = np.zeros( shape = ( np.shape( interface_point_pair )[0], 1) )
    for i in range( np.shape( interface_point_pair )[0] ):
        if interface_point_pair[i, 2] != 0 and interface_point_pair[i, 3] != 0:
            Lintface[i, interface_point_pair[i, 0]] =  1
            Lintface[i, interface_point_pair[i, 1]] =  1
            Lintface[i, interface_point_pair[i, 2]] =  -1
            Lintface[i, interface_point_pair[i, 3]] =  -1
        else:
            Lintface[i, ui.function_space.num_dof + uo.function_space.num_dof + interface_point_pair[i, 0]] =  1
            Lintface[i, ui.function_space.num_dof + uo.function_space.num_dof + interface_point_pair[i, 1]] = -1
    
#    Lintface = sparse.csr_matrix(Lintface)
    # %% 
    #Lintface=Lintface[~np.all(Lintface == 0, axis=1)]
    
    LHS = sparse.vstack( (LHS1, LHS2, LHS3, Lintface, LBC) )
    RHS = np.vstack( (RHS1, RHS2, RHS3, Rintface, RBC) )
    #rrefLHS = np.array(Matrix(LHS).rref()[0]).astype(None)
    #print(rrefLHS)
    print("----------------------------------------------------")
    print("LHS shape:", np.shape(LHS))
    
    print("------ solve the square sparse system:......")
    LHS = sparse.csr_matrix(LHS)
    Res = sparse.linalg.spsolve(LHS,RHS)
#
#    print("------ solve the singular square sparse system:......")
#    solution = sparse.linalg.lsqr(LHS,RHS, atol=1e-20, btol=1e-20)
#    Res = solution[0].reshape((np.size(solution[0]),1))
#    residual = np.sum(np.abs(LHS.dot(Res) - RHS))
#    print("------ least square solution error =",residual)


#    print("++++++ solve the singular square full system:......")
#    solution= np.linalg.lstsq(LHS,RHS)

# %% eigen values and eigen vector
#    w, v = sp.linalg.eig(LHS.todense())
    
    # %%
    ui.cochain = Res[0:ui.function_space.num_dof].reshape(ui.function_space.num_dof)
    uo.cochain = Res[ui.function_space.num_dof : -p0.function_space.num_dof].reshape(uo.function_space.num_dof)
    p0.cochain = Res[-p0.function_space.num_dof:].reshape(p0.function_space.num_dof)
    
    # %% view the result
    p0.reconstruct(xi, eta)
    (x, y), data = p0.export_to_plot()
    plt.contourf(x, y, data)
    plt.title('solution lobatto 0-form, p0')
    plt.colorbar()
    plt.show()
    print('p0 max:', np.max(data))
    print('p0 min:', np.min(data))
    
    uo.reconstruct(xi, eta)
    (x, y), data_dx, data_dy = uo.export_to_plot()
    plt.contourf(x, y, data_dx)
    plt.title('solution lobatto 1-form dx')
    plt.colorbar()
    plt.show()
    print('uo max:', np.max(data_dx))
    print('uo min:', np.min(data_dx))
    
    plt.contourf(x, y, data_dy)
    plt.title('solution lobatto 1-form dy')
    plt.colorbar()
    plt.show()
    print('uo max:', np.max(data_dy))
    print('uo min:', np.min(data_dy))
#    #
    ui.reconstruct(xi, eta)
    (x, y), data_dx, data_dy = ui.export_to_plot()
    plt.contourf(x, y, data_dx)
    plt.title('solution extended_gauss 1-form dx')
    plt.colorbar()
    plt.show()
    print('ui max:', np.max(data_dx))
    print('ui min:', np.min(data_dx))
    
    plt.contourf(x, y, data_dy)
    plt.title('solution extended_gauss 1-form dy')
    plt.colorbar()
    plt.show()
    print('ui max:', np.max(data_dy))
    print('ui min:', np.min(data_dy))
    
    f2_exact.reconstruct(xi, eta)
    (x, y), data = f2_exact.export_to_plot()
    plt.contourf(x, y, data)
    plt.title('exact extended-gauss 2-form, f2')
    plt.colorbar()
    plt.show()
    
# %% error 
    L2_error_p0 = p0.l_2_norm(pfun, ('gauss', 10))[0]
    print("------ L2_error_psi0 =", L2_error_p0)
    
    L2_error_ui = ui.l_2_norm((ui_dx, ui_dy), ('gauss', 5))[0]
    print("------ L2_error_ui =", L2_error_ui)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n")
# %% return
    return L2_error_p0, L2_error_ui
# %%
if __name__ == "__main__":
    i = 0
    for p in [15]:
        for c in [0]:
            for n in [1]:
                temp_L2_error_p0, temp_L2_error_ui = solver(p,n,c)              
                
                if i == 0:
                    p_c_n_l2p0_l2ui = np.array([p,c,n,temp_L2_error_p0,temp_L2_error_ui])
                else:
                    p_c_n_l2p0_l2ui = np.vstack((p_c_n_l2p0_l2ui, np.array([p,c,n,temp_L2_error_p0,temp_L2_error_ui])))
                i += 1 

#    scipy.io.savemat('curl_curl_h_convergence_0624_1.mat', mdict={'p_c_n_l2p0_l2ui': p_c_n_l2p0_l2ui})