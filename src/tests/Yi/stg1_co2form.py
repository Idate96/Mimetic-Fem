# -*- coding: utf-8 -*-
"""
(SHORT NAME EXPLANATION)
"""
import numpy as np
from mesh_crazy import CrazyMesh
import functionals
import matplotlib.pyplot as plt
from numpy.linalg import inv
from _assemble import assemble_

# %%
class STG1co2form(object):
    def __init__(self, mesh, p):
        self.mesh  = mesh
        assert isinstance(p, int) and p > 0, "p wrong, should be positive integer (px == py)."
        self.p = p
        
        self.k = 2
        self.is_coform = True
        
        self._cochain       = None
        self._cochain_local = None
        
        self._face_grid = 'extended_gauss', 'lobatto'
        self._face_nodes = [getattr(functionals, self._face_grid[i] +
                                    '_quad')(self.p)[0] for i in range(2)]

        self._default_quad_grid = ('gauss','gauss'), (self.p+2, self.p+2)
        self.quad_grid = None
    
        self.num_basis = 2* (self.p+1) * self.p
        self.num_basis_odx = self.num_basis_ody = (self.p+1) * self.p     
        
        self.func_odx         = self.func_ody         = None
        self.func_odx_in_form = self.func_ody_in_form = None
        
    # %%
    @property
    def func(self):
        return self.func_odx, self.func_ody
    
    @property
    def func_in_form(self):
        return self.func_odx_in_form, self.func_ody_in_form
    
    @func.setter
    def func(self,func):
        self.func_odx, self.func_ody = func
        self.func_odx_in_form = lambda x, y:  func[0](x,y)
        self.func_ody_in_form = lambda x, y: -func[1](x,y)
        
    # %% dof_map and related
    @property
    def dof_map(self):
        if isinstance(self.mesh, CrazyMesh):
            px = py = self.p
            nx = self.mesh.n_x
            ny = self.mesh.n_y
            local_numbering = np.array( [int(i) for i in range((px+1) * py + px * (py+1))] )
            global_numbering = np.zeros((nx * ny, (px+1) * py + px * (py+1) ), dtype=np.int64)
            for i in range(nx):
                for j in range(ny):
                    s = j + i * ny
                    global_numbering[s, :] = local_numbering + ((px+1) * py + px * (py+1)) * s
            return global_numbering
        
    @property
    def num_dof(self):
        return np.max(self.dof_map) + 1
    
    # %% cochain and related
    @property
    def cochain(self):
        return self._cochain
    
    @cochain.setter
    def cochain(self, cochain):
        try:
            assert np.shape(cochain) == (self.num_dof,)
            self._cochain = cochain
            self._cochain_local = self.cochain[np.transpose(self.dof_map)]
        except AssertionError:
            raise AssertionError(
                "The dofs do not match: cochain shape: {0}, number of degrees of freedom : {1}" 
                .format(np.shape(cochain), self.num_dof))
    
    @property
    def cochain_local(self):
        """Map the cochain elements into local dof with the dof map."""
        if self._cochain_local is None:
            assert self.cochain is not None, "cochain is empty, therefore no cochain_local"
            self._cochain_local = self.cochain[np.transpose(self.dof_map)]
        return self._cochain_local

    @cochain_local.setter
    def cochain_local(self, cochain):
        try:
            assert np.shape(cochain)[-1] == (self.mesh.num_elements)
            self._cochain_local = cochain
            self._cochain_to_global
        except AssertionError as e:
            raise AssertionError(
                "The number of the local cochain columns should be equal to the number of elements")
    @property
    def cochain_local_odx(self):
        return self.cochain_local[:self.num_basis_odx]
    
    @property
    def cochain_local_ody(self):
        return self.cochain_local[-self.num_basis_ody:]
    
    @property
    def _cochain_to_global(self):
        """Map the local dofs of the cochain into the global cochain."""
        self._cochain = np.zeros((self.num_dof))
        dof_map = np.transpose(self.dof_map)
        # reorder degrees of freedom
        for i, row in enumerate(self.cochain_local):
            for j, dof_value in enumerate(row):
                self._cochain[dof_map[i, j]] = dof_value
    
    # %%
    @property
    def quad_grid(self):
        return self._quad_type, self._quad_order
    
    @quad_grid.setter
    def quad_grid(self, quad ):
        if quad is None:
            self._quad_type, self._quad_order = self._default_quad_grid
        else:
            self._quad_type, self._quad_order = quad
            
        # init lists of nodes and weights
        self._quad_nodes = [0, 0]
        self._quad_weights = [0, 0]
        # create the nodes and the weights using quadrature methods
        self._quad_nodes[0], self._quad_weights[0] = getattr(
            functionals, self._quad_type[0] + '_quad')(self._quad_order[0])
        self._quad_nodes[1], self._quad_weights[1] = getattr(
            functionals, self._quad_type[1] + '_quad')(self._quad_order[1])
        
        if self._quad_type[0] == 'gauss':
            num_nodes_0 = self._quad_order[0]
        elif self._quad_type[0] == 'lobatto':
            num_nodes_0 = self._quad_order[0]+1
        else:
            raise("quad type wrong")
        if self._quad_type[1] == 'gauss':
            num_nodes_1 = self._quad_order[1]
        elif self._quad_type[1] == 'lobatto':
            num_nodes_1 = self._quad_order[1]+1  
        else:
            raise("quad type wrong")
        self._quad_nodes_num = (num_nodes_0, num_nodes_1)
        
    # %%
    def _inner_consistency_check(self, other):
        """Check consistency for inner product."""
        # chack mesh
        if self.mesh is not other.mesh:
            raise Exception("Mesh of the forms do not match")
            
    # %%  
    def evaluate_basis(self, domain=None):
        """Evaluate the basis."""
        if domain is None:
            edge_basis_1d_odx = [functionals.edge_basis(
                self._face_nodes[i], self._quad_nodes[i]) for i in range(2)]       
    
            edge_basis_1d_ody = [functionals.edge_basis(
                self._face_nodes[-i-1], self._quad_nodes[i]) for i in range(2)]  
    
            self.xi, self.eta = np.meshgrid(self._quad_nodes[0], self._quad_nodes[1])
        else:
            edge_basis_1d_odx = [functionals.edge_basis(
                self._face_nodes[i], domain[i]         ) for i in range(2)]
            edge_basis_1d_ody = [functionals.edge_basis(
                self._face_nodes[-i-1], domain[i]         ) for i in range(2)]
    
            self.xi, self.eta = np.meshgrid(*domain)

        self.basis_odx = np.kron(edge_basis_1d_odx[0], edge_basis_1d_odx[1])  
        self.basis_ody = np.kron(edge_basis_1d_ody[0], edge_basis_1d_ody[1])  
        self.basis = np.vstack((self.basis_odx, self.basis_ody))
    
    # %%
    def inner(self, other):
        self._inner_consistency_check(other)
        mesh = self.mesh
        quad_type, quad_order = 'gauss', np.max( [self.p, other.p] ) + 2
        quad_nodes, quad_weights = getattr(functionals, quad_type + '_quad')(quad_order)
        
        self.evaluate_basis(domain=(quad_nodes, quad_nodes))
        other.evaluate_basis(domain=(quad_nodes, quad_nodes))
        
        xi, eta = self.xi.ravel('F'), self.eta.ravel('F')
        
        M = np.einsum( 'ij,ljk->ilk', self.basis,
                                      np.einsum('lj,jk->ljk', other.basis, mesh.detinvJ(xi,eta)) )
        
        return M
    
    @property
    def M(self):
        M = self.inner(self)
        return M, assemble_(M, self.dof_map, self.dof_map)
        
    # %%
    def discretize(self, func, quad_type= None, quad_order=None ) :
        """func = (fun_odx, fun_ody), quad_grid = ( quad_type, quad_order ) """
        self.func = func
        
        if quad_type is not None:
            self.quad_grid = quad_type, quad_order
        else:
            self.quad_grid = ('gauss','gauss'), (self.p+5, self.p+5)
            
        dim_faces = [self._face_nodes[i][1:] - self._face_nodes[i][:-1]for i in range(2)]
        x_length = np.concatenate ((np.repeat(dim_faces[0],self.p), np.repeat(dim_faces[1],self.p+1)), axis = 0 )
        y_length = np.concatenate ((np.tile(dim_faces[1],self.p+1), np.tile(dim_faces[0],self.p))    , axis = 0 )

        g_area = 0.25 * x_length * y_length
        
        quad = np.meshgrid(self._quad_nodes[0], self._quad_nodes[1])
        quad_x = np.ravel(quad[0], order = 'C' )[np.newaxis, :].repeat(self.num_basis, axis = 0)
        quad_y = np.ravel(quad[1], order = 'C' )[np.newaxis, :].repeat(self.num_basis, axis = 0)
        
        quad_weights = np.meshgrid(self._quad_weights[0], self._quad_weights[1])
        quad_weights = (quad_weights[0] * quad_weights[1] ).ravel(order = 'C')

        bottom_left_point_xi = np.concatenate((np.repeat(self._face_nodes[0][:-1], self.p) ,
                                               np.repeat(self._face_nodes[1][:-1], self.p+1)),
                                               axis = 0)[:,np.newaxis].repeat(np.size(self._quad_nodes[0])*np.size(self._quad_nodes[1]),axis = 1)
        bottom_left_point_eta= np.concatenate((np.tile(self._face_nodes[1][:-1], self.p+1),
                                               np.tile(self._face_nodes[0][:-1], self.p)),
                                               axis = 0)[:,np.newaxis].repeat(np.size(self._quad_nodes[0])*np.size(self._quad_nodes[1]),axis = 1)
#        xi = ( bottom_left_point_xi.T  + ((quad_x + 1)/2).T * x_length ).T
#        eta= ( bottom_left_point_eta.T + ((quad_y + 1)/2).T * y_length ).T
        xi  = bottom_left_point_xi  + np.einsum('ij,i->ij',(quad_x + 1)/2, x_length) 
        eta = bottom_left_point_eta + np.einsum('ij,i->ij',(quad_y + 1)/2, y_length) 
        
        x, y = self.mesh.mapping(xi, eta)
        cochain_local_odx = np.einsum('ijk,j,i->ik',(self.func_in_form[0](x,y)[ :self.num_basis_odx] * self.mesh.dx_dxi(xi,eta)[:self.num_basis_odx]  + self.func_in_form[1](x,y)[ :self.num_basis_odx] * self.mesh.dy_dxi(xi,eta) [ :self.num_basis_odx]) * self.mesh.g(xi,eta)[ :self.num_basis_odx], quad_weights, g_area[ :self.num_basis_odx])
        cochain_local_ody = np.einsum('ijk,j,i->ik',(self.func_in_form[0](x,y)[-self.num_basis_ody:] * self.mesh.dx_deta(xi,eta)[self.num_basis_odx:] + self.func_in_form[1](x,y)[-self.num_basis_ody:] * self.mesh.dy_deta(xi,eta)[-self.num_basis_ody:]) * self.mesh.g(xi,eta)[-self.num_basis_ody:], quad_weights, g_area[-self.num_basis_ody:])
        
        self.cochain_local = np.vstack((cochain_local_odx, cochain_local_ody))
        
        self.quad_grid = None
        
    
    # %% 
    def reconstruct(self, xi=None, eta=None, do_plot = True, do_return = False):
        """Lets reconstruct the STG1 2 form """
        
        assert self.cochain is not None, "no cochain to reconstruct"
        if xi is None and eta is None:
            xi  = eta = np.linspace( -1, 1, np.int(np.ceil(np.sqrt(10000 / self.mesh.num_elements))+1) )
        
        self.evaluate_basis(domain=(xi, eta))
        xi, eta = self.xi.ravel('F'), self.eta.ravel('F')

        # value: the value of the co-form at the reference domian, multpy them with metric to get the values on physical domain
        value_odx = np.tensordot(self.basis_odx, self.cochain_local_odx, axes=((0), (0)))
        value_ody = np.tensordot(self.basis_ody, self.cochain_local_ody, axes=((0), (0)))
        
        self.reconstructed_odx = self.mesh.dy_deta(xi,eta)/self.mesh.g(xi,eta)**2 * value_odx - self.mesh.dy_dxi (xi,eta)/self.mesh.g(xi,eta)**2 * value_ody
        self.reconstructed_ody =-self.mesh.dx_deta(xi,eta)/self.mesh.g(xi,eta)**2 * value_odx + self.mesh.dx_dxi (xi,eta)/self.mesh.g(xi,eta)**2 * value_ody
        
        x, y = self.mesh.mapping(self.xi, self.eta)
        if do_return is True:
            return (x, y), self.reconstructed_odx, self.reconstructed_ody
        
        if do_plot is True:
            if isinstance(self.mesh, CrazyMesh):
                num_pts_y, num_pts_x = np.shape(self.xi)
                num_el_x, num_el_y = self.mesh.n_x, self.mesh.n_y
    
                x_4d = x.reshape(num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
                x = np.moveaxis(x_4d, 2, 1).reshape(
                    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')
    
                y_4d = y.reshape(num_pts_y, num_pts_x,  num_el_y, num_el_x, order='F')
                y = np.rollaxis(y_4d, 2, 1).reshape(
                    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')
    
                recon_4d_dx = self.reconstructed_odx.reshape(
                    num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
                reconstructed_odx = np.moveaxis(recon_4d_dx, 2, 1).ravel('F').reshape(
                    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')
    
                recon_4d_dy = self.reconstructed_ody.reshape(
                    num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
                reconstructed_ody = np.moveaxis(recon_4d_dy, 2, 1).ravel('F').reshape(
                    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')
            
                plt.contourf(x, y, reconstructed_odx)
                plt.title("STG1 co2-form odx")
                plt.colorbar()
                plt.show()
                
                plt.contourf(x, y, -reconstructed_ody)
                plt.title("STG1 co2-form ody")
                plt.colorbar()
                plt.show()

    # %%    
    def L2_error(self, func = None):
        if func is not None:
            self.func = func
        assert self.func != (None, None), "no exact func do not exist, no L2_error"

# %%
if __name__ == '__main__':
    crazy_mesh = CrazyMesh(2, (20, 20), ((-1, 1), (-1, 1)), curvature = 0.15)
    f2 = STG1co2form(crazy_mesh,5)
    
    def f_odx(x,y): return np.sin(np.pi*x) * np.cos(np.pi*y)
    def f_ody(x,y): return np.cos(np.pi*x) * np.sin(np.pi*y)
    
    f2.discretize((f_odx, f_ody))
    
    f2.reconstruct()
    
    
    