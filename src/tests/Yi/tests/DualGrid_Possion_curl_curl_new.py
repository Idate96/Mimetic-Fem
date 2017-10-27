import path_magic
from function_space import FunctionSpace
import numpy as np
import numpy.testing as npt
from mesh import CrazyMesh
from forms import Form
from basis_forms import BasisForm
from hodge import hodge
from coboundaries import d
from hodge import hodge
from inner_product import inner
from _assembling import assemble
import matplotlib.pyplot as plt
from quadrature import extended_gauss_quad
from scipy.integrate import quad
from tqdm import *
from sympy import Matrix
import scipy.io
from scipy import sparse
# %%
def pfun(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def ufun_u(x,y):
	return -np.pi * np.sin(np.pi*x) * np.cos(np.pi*y)

def ufun_v(x,y):
	return np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)

def ffun(x, y):
    return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# %%
def solver(p,n,c):
    # %% define
    px = py = p
    nx = n
    ny = n
    mesh = CrazyMesh(2, (nx, ny), ((-1, 1), (-1, 1)), c)
    xi = eta = np.linspace(-1, 1, np.ceil(1000 / (nx * ny)) + 1)
    
    
    # %% exact p(0)
    func_space_gl0 = FunctionSpace(mesh, '0-lobatto', (px + 1, py + 1), is_inner=False)
#    p0_exact = Form(func_space_gl0)
#    p0_exact.discretize(pfun)
#    p0_exact.reconstruct(xi, eta)
#    (x, y), data = p0_exact.export_to_plot()
#    plt.contourf(x, y, data)
#    plt.title('exact lobatto 0-form, p0')
#    plt.colorbar()
#    plt.show()
    
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
    E10_assembled = assemble(mesh, E10, uo.function_space.dof_map.dof_map,
                             p0.function_space.dof_map.dof_map, mode='replace')
    
    
    H = hodge(func_space_gl1)
    H_assembled   = assemble(mesh, H  , ui.function_space.dof_map.dof_map_internal, uo.function_space.dof_map.dof_map)
    #H_assembled   = np.linalg.inv(H_assembled)
    
    
    E21 = d(func_space_eg1)
    E21_assembled = assemble(mesh, E21, f2.function_space.dof_map.dof_map, ui.function_space.dof_map.dof_map, mode = 'replace')
    
    # %%
    #uo.cochain = E10_assembled.dot(p0_exact.cochain)
    #uo.reconstruct(xi, eta)
    #(x, y), data_dx, data_dy = uo.export_to_plot()
    #plt.contourf(x, y, data_dx)
    #plt.title('test lobatto outer 1-form dx')
    #plt.colorbar()
    #plt.show()
    #
    #plt.contourf(x, y, data_dy)
    #plt.title('test lobatto outer 1-form dy')
    #plt.colorbar()
    #plt.show()
    #
    #cochain_internal = H_assembled.dot(uo.cochain)
    #ui.cochain = np.concatenate((cochain_internal, np.zeros(ui.function_space.num_dof - ui.basis.num_basis * ui.mesh.num_elements )), axis=0)
    #ui.reconstruct(xi, eta)
    #(x, y), data_dx, data_dy = ui.export_to_plot()
    #plt.contourf(x, y, data_dx)
    #plt.title('test extended gauss inner 1-form dx')
    #plt.colorbar()
    #plt.show()
    #
    #plt.contourf(x, y, data_dy)
    #plt.title('test extended gauss inner 1-form dy')
    #plt.colorbar()
    #plt.show()
    # %%
    # system:
    #  |  I     H*E10   |   | ui |      | 0 |
    #  |                |   |    |  =   |   |
    #  | E21     0      |   | p  |      | f |
    ui_num_dof_internal = ui.basis.num_basis * ui.mesh.num_elements
    ui_num_dof_external = ui.function_space.num_dof - ui_num_dof_internal
    
    LHS1 = np.hstack((   np.eye(ui_num_dof_internal) , 
                         np.zeros((ui_num_dof_internal, ui_num_dof_external)),
                        -H_assembled.dot(E10_assembled)    ))
    
    LHS3 = np.hstack((  E21_assembled , 
                        np.zeros((f2.function_space.num_dof, p0.function_space.num_dof ))    ))

    
    RHS1 = np.zeros((ui_num_dof_internal, 1 ))
    RHS3 = f2_exact.cochain.reshape( (f2_exact.function_space.num_dof,1) )
    
    LHS3[-1,:]  = 0
    LHS3[-1,-1] = 1
    RHS3[-1,0]  = 0
    # %% boundary edges
    def UBC(mesh, s, p, position):
        def pullbackedfun_dx(xi, eta):
            x, y = mesh.mapping(xi, eta, s)
            return ufun_u(x, y)
    
        def pullbackedfun_dy(xi, eta):
            x, y = mesh.mapping(xi, eta, s)
            return ufun_v(x, y)
    
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
    
    Left, Right, Bottom, Top = extended_gauss1_general_boundary_edges(mesh, px, ui.function_space.dof_map.dof_map)
    Boundaryedgs = np.hstack( (Left, Right, Bottom, Top) )
    
    LBC = np.zeros( shape = ( np.shape(Boundaryedgs)[1], ui.function_space.num_dof + p0.function_space.num_dof ) )
    RBC = np.zeros( shape = ( np.shape(Boundaryedgs)[1], 1) )
    for i in range(np.shape(Boundaryedgs)[1]):
        LBC[i, int(Boundaryedgs[0, i])] =1
        RBC[i] = Boundaryedgs[1, i]
    #ui.cochain = np.concatenate((cochain_internal, Bottom[1,:], Top[1,:], Left[1,:], Right[1,:]))
    #f2.cochain = E21_assembled.dot(ui.cochain)
    #f2.discretize(ffun)
    #f2.reconstruct(xi, eta)
    #(x, y), data = f2.export_to_plot()
    #plt.contourf(x, y, data)
    #plt.title('test extended-gauss 2-form, f2')
    #plt.colorbar()
    #plt.show()
    
    #size_Left = np.size( Left ) /2
    # %% 
    def dof_map_crazy_lobatto_point_pair(mesh, p, global_numbering):
        nx, ny = mesh.n_x, mesh.n_y
        
        N = p+1
        
        interface_point_pair = np.zeros( ( ( (nx - 1) * ny + nx * (ny - 1) ) * N, 2 ), dtype=np.int32 )
        
        n = 0
        
        for i in range(nx - 1):
            for j in range(ny):
                s1 = j + i * ny
                s2 = j + (i + 1) * ny
    #            print(s1, s2)
                for m in range(N):
                    interface_point_pair[n, 0] = global_numbering[s1, N * p + m]
                    interface_point_pair[n, 1] = global_numbering[s2,         m]
                    
    #                if j < ny-1 and m == N-1:
    #                    interface_point_pair[n, 0] = interface_point_pair[n, 1] = 0
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
    interface_point_pair = dof_map_crazy_lobatto_point_pair(mesh, px+1, p0.function_space.dof_map.dof_map)
    
    Lintface = np.zeros( shape = ( np.shape( interface_point_pair )[0], ui.function_space.num_dof + p0.function_space.num_dof ) ) 
    #Rintface = np.zeros( shape = ( np.shape( interface_point_pair )[0]-(nx-1)*(ny-1), 1) )
    Rintface = np.zeros( shape = ( np.shape( interface_point_pair )[0], 1) )
    for i in range( np.shape( interface_point_pair )[0] ):
        if interface_point_pair[i, 0] == interface_point_pair[i, 1] == 0:
            pass
        else:
            Lintface[i, ui.function_space.num_dof + interface_point_pair[i, 0]] =  1
            Lintface[i, ui.function_space.num_dof + interface_point_pair[i, 1]] = -1
    
    # %% 
    #Lintface=Lintface[~np.all(Lintface == 0, axis=1)]
    LHS = np.vstack( (LHS1, LHS3, Lintface, LBC) )
    RHS = np.vstack( (RHS1, RHS3, Rintface, RBC) )
    #rrefLHS = np.array(Matrix(LHS).rref()[0]).astype(None)
    #print(rrefLHS)
#    solution= np.linalg.lstsq(LHS,RHS)
    
    sLHS = sparse.csr_matrix(LHS)
    solution = sparse.linalg.lsqr(sLHS,RHS)

    Res = solution[0].reshape((np.size(solution[0]),1))
    residual = np.sum(np.abs(LHS.dot(Res) - RHS))
    print("residual=",residual)
    
    # %%
    ui.cochain = Res[0:ui.function_space.num_dof].reshape(ui.function_space.num_dof)
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
    
    L2_error_p0 = p0.l_2_norm(pfun, ('gauss', 40))[0]
    print(L2_error_p0)
    
    L2_error_ui = ui.l_2_norm((ufun_u, ufun_v), ('lobatto', 40))[0]
    print(L2_error_ui)
    
    return L2_error_p0, L2_error_ui
if __name__ == "__main__":
    # %%
    i = 0
    
    n = 2
    for c in [0]:
        for p in [6 ]:
            temp_L2_error_p0, temp_L2_error_ui = solver(p,n,c)
            if i == 0:
                Result = np.array([p,c,n,temp_L2_error_p0,temp_L2_error_ui])
            else:
                Result = np.vstack((Result, np.array([p,c,n,temp_L2_error_p0,temp_L2_error_ui])))
            i += 1
    
    scipy.io.savemat('curl_curl_p_convergence_1.mat',
			mdict={'p_c_n_L2p0_L2ui': Result})