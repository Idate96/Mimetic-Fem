
from function_space import FunctionSpace
import numpy as np
from mesh import CrazyMesh
from forms import Form
from hodge import hodge
from coboundaries import d
from _assembling import assemble_
from assemble import assemble
import matplotlib.pyplot as plt
import scipy.io
from scipy import sparse

# %% define the exact solution
def pfun(x, y):
    return np.sin(2*np.pi * x) * np.sin(2*np.pi * y)

def uo_dy(x, y):
    return 2*np.pi * np.cos(2*np.pi * x) * np.sin(2*np.pi * y)

def uo_dx(x, y):
    return -2*np.pi * np.sin(2*np.pi * x) * np.cos(2*np.pi * y)

def ffun(x, y):
    return -8 * np.pi**2 * np.sin(2*np.pi * x) * np.sin(2*np.pi * y)
# %%
def solver(p,n,c):
    px = py = p
    nx = ny = n
    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("Start div grad solver @ p=", px)
    print("                      @ n=", nx)
    print("                      @ c=", c )
    mesh = CrazyMesh(2, (nx, ny), ((0, 1), (0, 1)), c)
    xi = eta = np.linspace(-1, 1, np.ceil(500 / (nx * ny)) + 1)
    
    # %% function spaces
    func_space_eg0 = FunctionSpace(mesh, '0-ext_gauss', (px, py))
    p0 = Form(func_space_eg0)
    p0_exact = Form(func_space_eg0)
    p0_exact.discretize(pfun)
    #p0_exact.reconstruct(xi, eta)
    #(x, y), data = p0_exact.export_to_plot()
    #plt.contourf(x, y, data)
    #plt.title('exact extended gauss 0-form, p0')
    #plt.colorbar()
    #plt.show()
    
    func_space_eg1 = FunctionSpace(mesh, '1-ext_gauss', (px, py))
    ui = Form(func_space_eg1)
    
    func_space_gl1 = FunctionSpace(mesh, '1-lobatto', (px + 1, py + 1), is_inner=False)
    uo = Form(func_space_gl1)
    
    func_space_gl2 = FunctionSpace(mesh, '2-lobatto', (px + 1, py + 1), is_inner=False)
    f2 = Form(func_space_gl2)
    f2.discretize(ffun, ('gauss', px+5))
    #    f2.reconstruct(xi, eta)
    #    (x, y), data = f2.export_to_plot()
    #    plt.contourf(x, y, data)
    #    plt.title('exact lobatto 2-form, f2')
    #    plt.colorbar()
    #    plt.show()
    
    # %%
    E10 = d(func_space_eg0)[0:ui.basis.num_basis]
    #    E10 = d(func_space_eg0)
    H = hodge(func_space_gl1)
    E21 = d(func_space_gl1)
     
    #     print(np.shape(uo.function_space.dof_map.dof_map))
    #     print(np.shape(ui.function_space.dof_map.dof_map_internal))
    
#    E10_assembled = assemble_(mesh, E10, ui.function_space.dof_map.dof_map_internal,
#                             p0.function_space.dof_map.dof_map, mode='replace') 
    E10_assembled = assemble(E10, (ui.function_space, p0.function_space) )
    
    #    H_assembled = assemble_(mesh, H, uo.function_space.dof_map.dof_map,
    #                           ui.function_space.dof_map.dof_map_internal)
    
#    E21_assembled = assemble_(mesh, E21, f2.function_space.dof_map.dof_map,
#                             uo.function_space.dof_map.dof_map, mode='replace')
    E21_assembled = assemble(E21, (f2.function_space,uo.function_space) )
    
    #print(E21_assembled)
    #    E10_assembled = assemble(E10, ui.function_space, p0.function_space) 
    H_assembled = -assemble(H, ui.function_space,uo.function_space)
    #    E21_assembled = assemble(E21, f2.function_space, uo.function_space)
    
    # %% test the assembled matrices
    #    ui_cochian_internal = E10_assembled.dot(p0_exact.cochain)
    #    ui.cochain = np.concatenate((ui_cochian_internal, np.zeros(
    #        ui.function_space.num_dof - ui.basis.num_basis * ui.mesh.num_elements)), axis=0)
    #    ui.reconstruct(xi, eta)
    #    (x, y), data_dx, data_dy = ui.export_to_plot()
    ##    plt.contourf(x, y, data_dx)
    ##    plt.title('exact extended_gauss 1-form dx')
    ##    plt.colorbar()
    ##    plt.show()
    #    
    #    plt.contourf(x, y, data_dy)
    #    plt.title('exact extended_gauss 1-form dy')
    #    plt.colorbar()
    #    plt.show()
    #    
    #    uo.cochain = H_assembled.dot( ui_cochian_internal)
    #    uo.reconstruct(xi, eta)
    #    (x, y), data_dx, data_dy = uo.export_to_plot()
    #    plt.contourf(x, y, data_dx)
    #    plt.title('exact lobbat 1-form dx')
    #    plt.colorbar()
    #    plt.show()
    
    #    plt.contourf(x, y, data_dy)
    #    plt.title('exact lobbat 1-form dy')
    #    plt.colorbar()
    #    plt.show()
    
    # %%
    # system:
    #  |  H      E10 |   | uo |      | 0 |
    #  |             |   |    |      |   |
    #  |             | * |    |   =  |   |
    #  |             |   |    |      |   |
    #  |  E21     0  |   | p  |      | f |
    ui_num_dof_internal = ui.basis.num_basis * ui.mesh.num_elements
    
    #    LHS1 = np.hstack((np.eye(ui_num_dof_internal),
    #                      np.zeros((ui_num_dof_internal, uo.function_space.num_dof)),
    #                      -E10_assembled))
    #    
    #    LHS2 = np.hstack((-H_assembled,
    #                      np.eye(uo.function_space.num_dof),
    #                      np.zeros((uo.function_space.num_dof, p0.function_space.num_dof))))
    #    
    #    LHS3 = np.hstack((np.zeros((f2.function_space.num_dof, ui_num_dof_internal)),
    #                      E21_assembled,
    #                      np.zeros((f2.function_space.num_dof, p0.function_space.num_dof))))
    
    LHS1 = sparse.hstack(( H_assembled,
                           E10_assembled ))
    
    LHS2 = sparse.hstack(( E21_assembled,
                           sparse.csc_matrix((f2.function_space.num_dof, p0.function_space.num_dof)) ))
    
    RHS1 = np.zeros(shape=(uo.function_space.num_dof, 1))
    RHS2 = f2.cochain.reshape((f2.function_space.num_dof, 1))
    
    # %%
    def dof_map_crazy_lobatto_edges(mesh, p):
        nx, ny = mesh.n_x, mesh.n_y
        global_numbering = np.zeros((nx * ny, 2 * p * (p + 1)), dtype=np.int32)
        local_numbering = np.array([int(i) for i in range(2 * p * (p + 1))])
        for i in range(nx):
            for j in range(ny):
                s = j + i * ny
                global_numbering[s, :] = local_numbering + 2 * p * (p + 1) * s
        interface_edge_pair = np.zeros((((nx - 1) * ny + nx * (ny - 1)) * p, 2), dtype=np.int32)
        n = 0
        for i in range(nx - 1):
            for j in range(ny):
                s1 = j + i * ny
                s2 = j + (i + 1) * ny
                for m in range(p):
                    interface_edge_pair[n, 0] = global_numbering[s1, p * (p + 1) + p**2 + m]
                    interface_edge_pair[n, 1] = global_numbering[s2, p * (p + 1) + m]
                    n += 1
        for i in range(nx):
            for j in range(ny - 1):
                s1 = j + i * ny
                s2 = j + 1 + i * ny
                for m in range(p):
                    interface_edge_pair[n, 0] = global_numbering[s1, (m + 1) * (p + 1) - 1]
                    interface_edge_pair[n, 1] = global_numbering[s2,  m * (p + 1)]
                    n += 1
        return interface_edge_pair
    
    interface_edge_pair = dof_map_crazy_lobatto_edges(mesh, px+1)
    LItFuo = sparse.lil_matrix(( np.shape( interface_edge_pair )[0], uo.function_space.num_dof + p0.function_space.num_dof ) ) 
    RItFuo = np.zeros( shape = ( np.shape( interface_edge_pair )[0], 1) )
    for i in range( np.shape( interface_edge_pair )[0] ):
        LItFuo[i, interface_edge_pair[i, 0]] =  1
        LItFuo[i, interface_edge_pair[i, 1]] = -1
        
    # %%    
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
                                    mesh, px, p0.function_space.dof_map.dof_map)
    Boundarypoint = np.hstack((Left, Right, Bottom, Top))
    #    LBCphi = np.zeros(shape=(np.size(Boundarypoint), ui_num_dof_internal +
    #                             uo.function_space.num_dof + p0.function_space.num_dof))
    LBCphi = sparse.lil_matrix((np.size(Boundarypoint), uo.function_space.num_dof + p0.function_space.num_dof))
    RBCphi = np.zeros(shape=(np.size(Boundarypoint), 1))
    for i in range(np.size(Boundarypoint)):
        LBCphi[i, uo.function_space.num_dof + Boundarypoint[i]] = 1
        RBCphi[i] = p0_exact.cochain[Boundarypoint[i]]
    
    # %% LHS and RHS and solve it
    LHS = sparse.vstack((LHS1, LHS2, LItFuo, LBCphi))
    RHS =     np.vstack((RHS1, RHS2, RItFuo, RBCphi))
    print("----------------------------------------------------")
    print("LHS shape:", np.shape(LHS))
    #    
    LHS = sparse.csr_matrix(LHS)
    print("------ solve the square sparse system:......")
    Res = sparse.linalg.spsolve(LHS,RHS)
    #    Res = sparse.linalg.lsqr(LHS,RHS, atol=1e-20, btol=1e-20)[0]
    
    #    print("++++++ solve the singular square full system:......")
    #    Res = np.linalg.solve(LHS, RHS)
    
    # %% split into pieces
    uo.cochain = Res[:uo.function_space.num_dof ].reshape(uo.function_space.num_dof)
    p0.cochain = Res[-p0.function_space.num_dof:].reshape(p0.function_space.num_dof)
    
    # %% view the result
    p0.reconstruct(xi, eta)
    (x, y), data = p0.export_to_plot()
    plt.contourf(x, y, data)
    plt.title('solution extended gauss 0-form, p0')
    plt.colorbar()
    plt.show()
    #    
#    uo.reconstruct(xi, eta)
#    (x, y), data_dx, data_dy = uo.export_to_plot()
#    plt.contourf(x, y, data_dx)
#    plt.title('lobatto 1-form dx')
#    plt.colorbar()
#    plt.show()
#    
#    plt.contourf(x, y, data_dy)
#    plt.title('lobatto 1-form dy')
#    plt.colorbar()
#    plt.show()
    
    #f2.reconstruct(xi, eta)
    #(x, y), data = f2.export_to_plot()
    #plt.contourf(x, y, data)
    #plt.title('exact lobatto 2-form, f2')
    #plt.colorbar()
    #plt.show()
    
    # %% L2_error
    L2_error_p0 = p0.l_2_norm(pfun, ('gauss', px + 5))[0]
    
    L2_error_uo = uo.l_2_norm((uo_dx, uo_dy), ('lobatto', px + 5))[0]
    
    print("------ L2_error_p0 =", L2_error_p0)
    print("------ L2_error_uo =", L2_error_uo)
    return L2_error_p0, L2_error_uo
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n")
# %%
if __name__ == "__main__":
    i = 0
    for n in [3]:
        for c in [0, 0.3]:
            for p in [2,4,6,8,10,12,14,16,18,20,22,24]:
                temp_L2_error_p0, temp_L2_error_uo = solver(p,n,c)
                if i == 0:
                    p_c_n_l2p0_l2uo = np.array( [p,c,n,temp_L2_error_p0,temp_L2_error_uo] )
                else:
                    p_c_n_l2p0_l2uo = np.vstack((p_c_n_l2p0_l2uo, np.array([p,c,n,temp_L2_error_p0,temp_L2_error_uo])))
                i += 1 

    scipy.io.savemat('div_grad_p_convergence_0703_yi_n3.mat', mdict={'p_c_n_l2p0_l2uo': p_c_n_l2p0_l2uo})