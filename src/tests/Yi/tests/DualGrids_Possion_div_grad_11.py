from function_space import FunctionSpace
import numpy as np
from mesh import CrazyMesh
from forms import Form
from hodge import hodge
from coboundaries import d
from _assembling import assemble_, integral1d_
from assemble import assemble
import matplotlib.pyplot as plt
import scipy.io
from scipy import sparse
from inner_product import inner

# %% define the exact solution
def pfun(x, y):
    return np.sin(2*np.pi * x) * np.sin(2*np.pi * y)

def uo_dy(x, y):
    return 2*np.pi * np.cos(2*np.pi * x) * np.sin(2*np.pi * y)

def uo_dx(x, y):
    return -2*np.pi * np.sin(2*np.pi * x) * np.cos(2*np.pi * y)

def ffun(x, y):
    return -8 * np.pi**2 * np.sin(2*np.pi * x) * np.sin(2*np.pi * y)

#def solver(p,n,c):
p = 2
n = 2
c = 0.1

# %% define
px = py = p
nx = ny = n
print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("Start div grad solver @ p=", px)
print("                      @ n=", nx)
print("                      @ c=", c)
mesh = CrazyMesh(2, (nx, ny), ((0, 1), (0, 1)), c)
xi = eta = np.linspace(-1, 1, np.ceil(500 / (nx * ny)) + 1)
# %% function spaces
func_space_eg0 = FunctionSpace(mesh, '0-ext_gauss', (px, py))
p0       = Form(func_space_eg0)
p0_exact = Form(func_space_eg0)
p0_exact.discretize(pfun)
#    p0_exact.reconstruct(xi, eta)
#    (x, y), data = p0_exact.export_to_plot()
#    plt.contourf(x, y, data)
#    plt.title('exact extended gauss 0-form, p0')
#    plt.colorbar()
#    plt.show()
#    func_space_g0 = FunctionSpace(mesh, '0-gauss', (px, py))
# %%
func_space_eg1 = FunctionSpace(mesh, '1-ext_gauss', (px, py))
ui = Form(func_space_eg1)
# %%
func_space_gl1 = FunctionSpace(mesh, '1-lobatto', (px + 1, py + 1), is_inner=False)
uo = Form(func_space_gl1)
# %%
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
#E10           = d(func_space_eg0)[0:ui.basis.num_basis]
#E10_assembled = assemble_(mesh, E10, ui.function_space.dof_map.dof_map_internal,
#                         p0.function_space.dof_map.dof_map, mode='replace') 
##    E10 = d(func_space_eg0)
#
#
#Wn0 = p0.basis.wedged(f2.basis)
#Wn0_assembled = assemble_(mesh, Wn0, f2.function_space.dof_map.dof_map,
#                          p0.function_space.dof_map.dof_map_internal,mode='add') 
#
#Mn           = inner   (    f2.basis,          f2.basis)
#Mn_assembled = assemble(Mn, f2.function_space, f2.function_space)
#
#M0           = inner   (    p0.basis         , p0.basis)
#M0_assembled = assemble(M0, p0.function_space, p0.function_space)
#
#H11           = hodge    (func_space_gl1)
#H11_assembled = -assemble(H11, ui.function_space,uo.function_space)

#ui_num_dof_internal = ui.basis.num_basis * ui.mesh.num_elements
p0_num_dof_internal = p0.basis.num_basis * ui.mesh.num_elements
# %% LHS 11
Mnm1           = inner   (      uo.basis         , uo.basis)
Mnm1_assembled = assemble(Mnm1, uo.function_space, uo.function_space)

# %% LHS 21
W0n           = f2.basis.wedged(p0.basis)
#    W0n_assembled = assemble_(mesh, W0n, p0.function_space.dof_map.dof_map_internal,
#                              f2.function_space.dof_map.dof_map,mode='add') 
W0n_assembled = assemble(W0n, (p0.function_space,f2.function_space) )
E21           = d(func_space_gl1)
#E21_assembled = assemble_(mesh, E21, f2.function_space.dof_map.dof_map,
#                          uo.function_space.dof_map.dof_map, mode='replace')
LHS21_local = W0n.dot(E21) 
LHS12_local = LHS21_local.T 

LHS12_add   = sparse.lil_matrix(( np.shape(LHS21_local)[1], (px+1)*4 ))
Wb = integral1d_(1, ('lobatto_edge',px+1), ('gauss_node', px+1), ('gauss', px+5))
P = (p+1) * (p+2)
Q = (p+1)**2
M =  p+1
for i in range(p+1):
    for j in range(p+1):
        # left
        LHS12_add[P+i           ,      j] =  - Wb[i,j]
        # right
        LHS12_add[-p-1+i        ,  M + j] =  + Wb[i,j]
        # bottom
        LHS12_add[i*(p+2)       , 2*M+ j] =  - Wb[i,j]
        # top
        LHS12_add[(i+1)*(p+2)-1 , 3*M+ j] =  + Wb[i,j]

LHS21_add = LHS12_add.T
LHS12_local = sparse.hstack((LHS12_local, LHS12_add))
LHS21_local = sparse.vstack((LHS21_local, LHS21_add))
#print(np.shape(LHS21_local))
#print(np.shape(LHS12_local))

#    LHS21= assemble_(mesh, LHS21_local.toarray(), p0.function_space.dof_map.dof_map,
#                              uo.function_space.dof_map.dof_map, mode='add')
LHS21= assemble(LHS21_local.toarray(), (p0.function_space, uo.function_space))

#    LHS12= assemble_(mesh, LHS12_local.toarray(), uo.function_space.dof_map.dof_map,
#                              p0.function_space.dof_map.dof_map, mode='add')
LHS12= assemble(LHS12_local.toarray(), (uo.function_space, p0.function_space))

#Hn0 =  sparse.linalg.inv(Mn_assembled).dot(Wn0_assembled)
#Hn0 =  sparse.linalg.inv(W0n_assembled).dot(M0_assembled)
#f2.cochain = Hn0.dot(p0_exact.cochain_internal)
#f2.reconstruct(xi, eta)
#(x, y), data = f2.export_to_plot()
#plt.contourf(x, y, data)
#plt.title('Hodge p0_exact into lobatto 2form')
#plt.colorbar()
#plt.show()
# %%
# system:
#  |  Mnm1   (W0n*E21)^T    |   | uo |      | 0     |
#  |                        |   |    |      |       |
#  |                        | * |    |   =  |       |
#  |                        |   |    |      |       |
#  |  W0n*E21     0         |   | p  |      | W0n*f |

LHS1 = sparse.hstack(( Mnm1_assembled,  LHS12 ))

LHS2 = sparse.lil_matrix(sparse.hstack(( LHS21, sparse.csc_matrix((p0.function_space.num_dof, p0.function_space.num_dof)))))

RHS1 = np.zeros(shape=(uo.function_space.num_dof, 1))
RHS2 = W0n_assembled.dot(f2.cochain.reshape((f2.function_space.num_dof, 1)))
RHS2add = np.zeros(shape=(p0.function_space.num_dof-f2.function_space.num_dof, 1))
RHS2 = np.vstack((RHS2, RHS2add))

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
#    Boundarypoint = np.hstack((Left, Right, Bottom, Top))
for j in range(np.size(Left)):
#    temp = np.sum(LHS2[Left[j], :])
    LHS2[Left[j], :] = 0
    LHS2[Left[j], uo.function_space.num_dof + Left[j]] = -1
    RHS2[Left[j]   ] = p0_exact.cochain[Left[j]]
    
    LHS2[Right[j], :] = 0
    LHS2[Right[j], uo.function_space.num_dof + Right[j]] = 1
    RHS2[Right[j]   ] =p0_exact.cochain[Right[j]]
    
    LHS2[Bottom[j], :] = 0
    LHS2[Bottom[j], uo.function_space.num_dof + Bottom[j]] = 1
    RHS2[Bottom[j]   ] = p0_exact.cochain[Bottom[j]]
    
    LHS2[Top[j], :] = 0
    LHS2[Top[j], uo.function_space.num_dof + Top[j]] = 1
    RHS2[Top[j]]    = p0_exact.cochain[Top[j]]
# %%
LHS = sparse.vstack((LHS1, LHS2))
RHS =     np.vstack((RHS1, RHS2))

# %%
print("----------------------------------------------------")
print("LHS shape:", np.shape(LHS))
#    
LHS = sparse.csr_matrix(LHS)
print("------ solve the square sparse system:......")
Res = sparse.linalg.spsolve(LHS,RHS)

# %% split into pieces
uo.cochain = Res[:uo.function_space.num_dof ].reshape(uo.function_space.num_dof)
#p0_cochain_internal = Res[-p0_num_dof_internal:].reshape(p0_num_dof_internal)
#p0.cochain = np.concatenate((p0_cochain_internal,np.zeros(p0.function_space.num_dof-p0_num_dof_internal)),axis=0)
p0.cochain = Res[-p0.function_space.num_dof:].reshape(p0.function_space.num_dof)
# %% plot the solution
p0.reconstruct(xi, eta)
(x, y), data_p0 = p0.export_to_plot()
plt.contourf(x, y, data_p0)
plt.title(r'solution extended gauss 0-form, $p^{(0)}$')
plt.colorbar()
plt.show()

#    uo.reconstruct(xi, eta)
#    (x, y), data_dx, data_dy = uo.export_to_plot()
#    plt.contourf(x, y, data_dx)
#    plt.title('solution lobatto 1-form dx')
#    plt.colorbar()
#    plt.show()
#    
#    plt.contourf(x, y, data_dy)
#    plt.title('solution lobatto 1-form dy')
#    plt.colorbar()
#    plt.show()

# %% L2_error
L2_error_p0 = p0.l_2_norm(pfun, ('gauss', px + 5))[0]
L2_error_uo = uo.l_2_norm((uo_dx, uo_dy), ('lobatto', px + 5))[0]
print("------ L2_error_p0 =", L2_error_p0)
print("------ L2_error_uo =", L2_error_uo)
#return L2_error_p0, L2_error_uo
print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n")

# %%
#if __name__ == "__main__":
#    i = 0
#    for n in [3]:
#        for c in [0, 0.3]:
#            for p in [2,4,6,8,10,12,14,16,18,20,22,24]:
#                temp_L2_error_p0, temp_L2_error_uo = solver(p,n,c)
#                if i == 0:
#                    p_c_n_l2p0_l2uo = np.array( [p,c,n,temp_L2_error_p0,temp_L2_error_uo] )
#                else:
#                    p_c_n_l2p0_l2uo = np.vstack((p_c_n_l2p0_l2uo, np.array([p,c,n,temp_L2_error_p0,temp_L2_error_uo])))
#                i += 1 
#
#    scipy.io.savemat('div_grad_p_convergence_0703_marc_n3.mat', mdict={'p_c_n_l2p0_l2uo': p_c_n_l2p0_l2uo})