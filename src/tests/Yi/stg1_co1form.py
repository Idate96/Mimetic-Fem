# -*- coding: utf-8 -*-
"""
(SHORT NAME EXPLANATION)
"""
import numpy as np
from mesh_crazy import CrazyMesh
import functionals
from _assemble import assemble_
import matplotlib.pyplot as plt
from numpy.linalg import inv
from stg1_co2form import STG1co2form
from lobatto_0form import Lobatto0form

# %%
class STG1co1form(object):
    def __init__(self, mesh, p, separated_dof = False):
        self.mesh  = mesh
        assert isinstance(p, int) and p > 0, "p wrong, should be positive integer, which means (px == py)."
        self.p = p
        
        self.k = 1
        self.dim = self.mesh.dim
        self.is_coform = True
        self.separated_dof = separated_dof
        
        self._cochain       = None
        self._cochain_local = None
        
        self._lobatto_nodes = getattr(functionals, 'lobatto' + '_quad')(self.p)[0] 
        self._ext_gauss_nodes  = getattr(functionals, 'extended_gauss'  + '_quad')(self.p)[0] 

        self._default_quad_grid = ('gauss','gauss'), (self.p+1, self.p+1)
        self.quad_grid = None
    
        self.num_basis = 2* ((self.p+1) ** 2  + self.p * (self.p + 2))
        self.num_basis_dxodx = self.num_basis_dyody = (self.p + 1)**2
        self.num_basis_dyodx = self.num_basis_dxody = self.p * (self.p + 2)
        
        self.func_dxodx = self.func_dyodx = self.func_dxody = self.func_dyody = None
        self.func_dxodx_in_form = self.func_dyodx_in_form = self.func_dxody_in_form = self.func_dyody_in_form = None
    
    # %%
    @property
    def func(self):
        return self.func_dxodx, self.func_dyodx, self.func_dxody, self.func_dyody
    @func.setter
    def func(self,func):
        self.func_dxodx, self.func_dyodx, self.func_dxody, self.func_dyody = func
        self.func_dxodx_in_form = lambda x, y: -func[0](x,y)
        self.func_dyodx_in_form = lambda x, y:  func[1](x,y)
        self.func_dxody_in_form = lambda x, y:  func[2](x,y)
        self.func_dyody_in_form = lambda x, y: -func[3](x,y) 

    @property
    def func_in_form(self):
        return self.func_dxodx_in_form, self.func_dyodx_in_form, self.func_dxody_in_form, self.func_dyody_in_form
    
    # %% dof_map and related
    @property
    def dof_map(self):
        if isinstance(self.mesh, CrazyMesh):
            nx, ny, p = self.mesh.n_x, self.mesh.n_y, self.p
            if self.separated_dof is True:
                # TODO:
                pass
            else:
                px, py = (p+1) * nx, p * ny
                global_numbering_dxodx_matrix = np.array( [ int(i) for i in range(px * (py+1)) ] ).reshape((py+1,px), order = 'F')
                global_numbering_dxodx        = np.zeros( shape=(nx*ny, (p+1)**2),dtype = np.int32 )
                
                global_numbering_dyodx_matrix = np.array( [ int(i) for i in range((px+1) * py) ] ).reshape((py, px + 1), order = 'F') + px * (py+1)
                global_numbering_dyodx        = np.zeros( shape=(nx*ny, p*(p+2)) ,dtype = np.int32 )
                              
                for i in range(nx):
                    for j in range(ny):
                        s = j + i * ny
                        global_numbering_dxodx[s,:] = global_numbering_dxodx_matrix[j*p:j*p+p+1,i*(p+1):(i+1)*(p+1)].ravel('F')
                        global_numbering_dyodx[s,:] = global_numbering_dyodx_matrix[j*p:(j+1)*p,i*(p+1):i*(p+1)+(p+2)].ravel('F')
               
                px, py = p * nx, (p+1) * ny
                global_numbering_dxody_matrix = np.array( [ int(i) for i in range( px * (py+1) ) ] ).reshape((py+1,px), order = 'F') + np.max(global_numbering_dyodx)+1
                global_numbering_dxody        = np.zeros( shape=(nx*ny, p*(p+2)) ,dtype = np.int32 )
                global_numbering_dyody_matrix = np.array( [ int(i) for i in range( (px+1) * py ) ] ).reshape((py,px+1), order = 'F') + np.max(global_numbering_dxody_matrix)+1
                global_numbering_dyody        = np.zeros( shape=(nx*ny, (p+1)**2) ,dtype = np.int32 )
                for i in range(nx):
                    for j in range(ny):
                        s = j + i * ny
                        global_numbering_dxody[s,:] = global_numbering_dxody_matrix[j*(p+1):j*(p+1)+(p+2),i*p:(i+1)*p].ravel('F')
                        global_numbering_dyody[s,:] = global_numbering_dyody_matrix[j*(p+1):(j+1)*(p+1),i*p:i*p+(p+1)].ravel('F')      

                return np.hstack( ( global_numbering_dxodx, global_numbering_dyodx, global_numbering_dxody,global_numbering_dyody ) )
        
    @property
    def num_dof(self):
        return np.max(self.dof_map) + 1
    
    @property
    def dof_map_interface_pairs(self):
        assert self.separated_dof is True, "not separated_dof, no interface_pairs"
        pass
    
    @property
    def dof_map_boundary(self):
        # TODO:
        pass
    
    # %% incidence matrix
    @property
    def coboundary(self):
        p = self.p
        E21 = np.zeros( shape = ( 2*p*(p+1), self.num_basis ) )
        for i in range(p):
            for j in range(p+1):
                num_2f = i + j * p
                edge_bottom = num_2f + j
                edge_top    = edge_bottom + 1
                edge_left   = self.num_basis_dxodx + num_2f
                edge_right  = edge_left + p
                
                E21[num_2f, edge_bottom] = -1
                E21[num_2f, edge_top]    = +1
                E21[num_2f, edge_left]   = +1
                E21[num_2f, edge_right]  = -1
                
                num_2f = j + i*(p+1)
                edge_bottom = self.num_basis_dxodx + self.num_basis_dyodx + num_2f + i
                edge_top    = edge_bottom + 1
                edge_left   = self.num_basis_dxodx + self.num_basis_dyodx + self.num_basis_dxody + num_2f
                edge_right  = edge_left + p+1
                num_2f += p*(p+1)
                E21[num_2f, edge_bottom] = -1
                E21[num_2f, edge_top]    = +1
                E21[num_2f, edge_left]   = +1
                E21[num_2f, edge_right]  = -1
        return -E21
    
    @property
    def coboundary_assembled(self):
        E21 = self.coboundary
        co2f = STG1co2form(self.mesh,self.p)
        E21_assembled = assemble_(E21, co2f.dof_map, self.dof_map)
        if self.cochain is not None:
            co2f.cochain = E21_assembled.dot(self.cochain)
        return E21_assembled, co2f
    
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
    def cochain_local_dxodx(self):
        return self.cochain_local[ : self.num_basis_dxodx]
    
    @property
    def cochain_local_dyodx(self):
        return self.cochain_local[ self.num_basis_dxodx : self.num_basis_dxodx + self.num_basis_dyodx]
    
    @property
    def cochain_local_dxody(self):
        return self.cochain_local[self.num_basis_dxodx + self.num_basis_dyodx : -self.num_basis_dyody]
    
    @property
    def cochain_local_dyody(self):
        return self.cochain_local[ -self.num_basis_dyody : ]
    
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
        # check mesh
        if self.mesh is not other.mesh:
            raise Exception("Mesh of the forms do not match")
            
    # %%  
    def evaluate_basis(self, domain=None):
        """Evaluate the basis."""
        pass
        if domain is None:
            edge_basis_eg = [functionals.edge_basis(self._ext_gauss_nodes, self._quad_nodes[i]) for i in range(2)]  
            edge_basis_gl = [functionals.edge_basis(self._lobatto_nodes  , self._quad_nodes[i]) for i in range(2)] 
            node_basis_eg = [functionals.lagrange_basis(self._ext_gauss_nodes, self._quad_nodes[i]) for i in range(2)] 
            node_basis_gl = [functionals.lagrange_basis(self._lobatto_nodes  , self._quad_nodes[i]) for i in range(2)] 
            self.xi, self.eta = np.meshgrid(self._quad_nodes, self._quad_nodes)
        else:
            edge_basis_eg = [functionals.edge_basis(self._ext_gauss_nodes, domain[i]) for i in range(2)]  
            edge_basis_gl = [functionals.edge_basis(self._lobatto_nodes  , domain[i]) for i in range(2)] 
            node_basis_eg = [functionals.lagrange_basis(self._ext_gauss_nodes, domain[i]) for i in range(2)] 
            node_basis_gl = [functionals.lagrange_basis(self._lobatto_nodes  , domain[i]) for i in range(2)] 
            self.xi, self.eta = np.meshgrid(*domain)

        self.basis_dxodx = np.kron(edge_basis_eg[0], node_basis_gl[1])  # OK
        self.basis_dyodx = np.kron(node_basis_eg[0], edge_basis_gl[1])  # OK
        self.basis_dxody = np.kron(edge_basis_gl[0], node_basis_eg[1])  # OK
        self.basis_dyody = np.kron(node_basis_gl[0], edge_basis_eg[1])  # OK
    
        self.basis = np.vstack( ( self.basis_dxodx, self.basis_dyodx, self.basis_dxody, self.basis_dyody ) )
    
    # %%
    def inner(self, other):
        self._inner_consistency_check(other)
        mesh = self.mesh
        quad_type, quad_order = 'gauss', np.max( [self.p, other.p] ) + 2
        quad_nodes, quad_weights = getattr(functionals, quad_type + '_quad')(quad_order)
        
        self.evaluate_basis(domain=(quad_nodes, quad_nodes))
        other.evaluate_basis(domain=(quad_nodes, quad_nodes))
        
        xi, eta = self.xi.ravel('F'), self.eta.ravel('F')
        
        b = np.vstack((np.einsum( 'ij,jk->ijk', other.basis_dxodx, mesh.invJ11(xi,eta) * mesh.invJ11(xi,eta) ),
                       np.einsum( 'ij,jk->ijk', other.basis_dyodx, mesh.invJ21(xi,eta) * mesh.invJ11(xi,eta) ),
                       np.einsum( 'ij,jk->ijk', other.basis_dxody, mesh.invJ11(xi,eta) * mesh.invJ21(xi,eta) ),
                       np.einsum( 'ij,jk->ijk', other.basis_dyody, mesh.invJ21(xi,eta) * mesh.invJ21(xi,eta) )  ))
        
        a = np.vstack((np.einsum( 'ij,jk->ijk', other.basis_dxodx, mesh.invJ12(xi,eta) * mesh.invJ11(xi,eta) ),
                       np.einsum( 'ij,jk->ijk', other.basis_dyodx, mesh.invJ22(xi,eta) * mesh.invJ11(xi,eta) ),
                       np.einsum( 'ij,jk->ijk', other.basis_dxody, mesh.invJ12(xi,eta) * mesh.invJ21(xi,eta) ),
                       np.einsum( 'ij,jk->ijk', other.basis_dyody, mesh.invJ22(xi,eta) * mesh.invJ21(xi,eta) )  ))
        
        d = np.vstack((np.einsum( 'ij,jk->ijk', other.basis_dxodx, mesh.invJ11(xi,eta) * mesh.invJ12(xi,eta) ),
                       np.einsum( 'ij,jk->ijk', other.basis_dyodx, mesh.invJ21(xi,eta) * mesh.invJ12(xi,eta) ),
                       np.einsum( 'ij,jk->ijk', other.basis_dxody, mesh.invJ11(xi,eta) * mesh.invJ22(xi,eta) ),
                       np.einsum( 'ij,jk->ijk', other.basis_dyody, mesh.invJ21(xi,eta) * mesh.invJ22(xi,eta) )  ))
        
        c = np.vstack((np.einsum( 'ij,jk->ijk', other.basis_dxodx, mesh.invJ12(xi,eta) * mesh.invJ12(xi,eta) ),
                       np.einsum( 'ij,jk->ijk', other.basis_dyodx, mesh.invJ22(xi,eta) * mesh.invJ12(xi,eta) ),
                       np.einsum( 'ij,jk->ijk', other.basis_dxody, mesh.invJ12(xi,eta) * mesh.invJ22(xi,eta) ),
                       np.einsum( 'ij,jk->ijk', other.basis_dyody, mesh.invJ22(xi,eta) * mesh.invJ22(xi,eta) )  ))
        
        M = np.vstack( (
        np.einsum( 'ij,ljk->ilk', self.basis_dxodx , ( -a * mesh.J12(xi,eta)*mesh.J11(xi,eta)+
                                                        b * mesh.J22(xi,eta)*mesh.J11(xi,eta)+
                                                       -c * mesh.J12(xi,eta)*mesh.J21(xi,eta)+
                                                        d * mesh.J22(xi,eta)*mesh.J21(xi,eta) ) )
        ,
        np.einsum( 'ij,ljk->ilk', self.basis_dyodx , (  a * mesh.J11(xi,eta)*mesh.J11(xi,eta)+
                                                       -b * mesh.J21(xi,eta)*mesh.J11(xi,eta)+
                                                        c * mesh.J11(xi,eta)*mesh.J21(xi,eta)+
                                                       -d * mesh.J21(xi,eta)*mesh.J21(xi,eta) ) )
        ,
        np.einsum( 'ij,ljk->ilk', self.basis_dxody , ( -a * mesh.J12(xi,eta)*mesh.J12(xi,eta)+
                                                        b * mesh.J22(xi,eta)*mesh.J12(xi,eta)+
                                                       -c * mesh.J12(xi,eta)*mesh.J22(xi,eta)+
                                                        d * mesh.J22(xi,eta)*mesh.J22(xi,eta) ) )
        ,
        np.einsum( 'ij,ljk->ilk', self.basis_dyody , (  a * mesh.J11(xi,eta)*mesh.J12(xi,eta)+
                                                       -b * mesh.J21(xi,eta)*mesh.J12(xi,eta)+
                                                        c * mesh.J11(xi,eta)*mesh.J22(xi,eta)+
                                                       -d * mesh.J21(xi,eta)*mesh.J22(xi,eta) ) )
        ) )
        return M
    
    @property
    def M(self):
        M = self.inner(self)
        return M, assemble_(M, self.dof_map, self.dof_map)
    
    # %%
    def discretize(self, func, quad_type= None, quad_order=None ) :
        """func = (fun_dxodx, fun_dyodx, fun_dxody, fun_dyody), quad_grid = ( quad_type, quad_order ) """
        self.func = func
        
        if quad_type is None:
            quad_type, quad_order = 'gauss', self.p+1
        
        quad_nodes, quad_weights = getattr(functionals, quad_type + '_quad')(quad_order)
        
        eg_edge_length = self._ext_gauss_nodes[1:] - self._ext_gauss_nodes[:-1]
        gl_edge_length = self._lobatto_nodes  [1:] - self._lobatto_nodes  [:-1]
        
        g_eg_length = 0.5 * eg_edge_length
        g_gl_length = 0.5 * gl_edge_length
        
        eg_start_point = self._ext_gauss_nodes[:-1].repeat(self.p+1,axis=0)
        xi_dxodx = eg_start_point[:,np.newaxis].repeat(np.size(quad_nodes),axis=1) + np.einsum('ij,i->ij',(quad_nodes[np.newaxis,:].repeat(self.num_basis_dxodx, axis=0) + 1)/2, eg_edge_length.repeat(self.p+1))
        eta_dxodx= np.tile(self._lobatto_nodes,self.p+1)[:,np.newaxis].repeat(np.size(quad_nodes),axis=1)
        
        gl_start_point = np.tile(self._lobatto_nodes[:-1],self.p+2)
        xi_dyodx = self._ext_gauss_nodes.repeat(self.p)[:,np.newaxis].repeat(np.size(quad_nodes), axis = 1)
        eta_dyodx= gl_start_point[:,np.newaxis].repeat(np.size(quad_nodes),axis=1) + np.einsum('ij,i->ij',(quad_nodes[np.newaxis,:].repeat(self.num_basis_dyodx, axis=0) + 1)/2, np.tile(gl_edge_length,self.p+2))
        
        gl_start_point = np.repeat(self._lobatto_nodes[:-1],self.p+2)
        xi_dxody = gl_start_point[:,np.newaxis].repeat(np.size(quad_nodes),axis=1) + np.einsum('ij,i->ij',(quad_nodes[np.newaxis,:].repeat(self.num_basis_dxody, axis=0) + 1)/2, gl_edge_length.repeat(self.p+2))
        eta_dxody= np.tile(self._ext_gauss_nodes,self.p)[:,np.newaxis].repeat(np.size(quad_nodes),axis=1)
        
        eg_start_point = np.tile(self._ext_gauss_nodes[:-1],self.p+1)
        xi_dyody = self._lobatto_nodes.repeat(self.p+1)[:,np.newaxis].repeat(np.size(quad_nodes), axis = 1)
        eta_dyody= eg_start_point[:,np.newaxis].repeat(np.size(quad_nodes),axis=1) + np.einsum('ij,i->ij',(quad_nodes[np.newaxis,:].repeat(self.num_basis_dyody, axis=0) + 1)/2, np.tile(eg_edge_length,self.p+1) )
        
        xi = np.vstack( (  xi_dxodx,  xi_dyodx,  xi_dxody,  xi_dyody ) )
        eta= np.vstack( ( eta_dxodx, eta_dyodx, eta_dxody, eta_dyody ) )
        
        x, y = self.mesh.mapping(xi ,eta)
        J11, J12, J21, J22 = self.mesh.dx_dxi(xi, eta), self.mesh.dx_deta(xi, eta), self.mesh.dy_dxi(xi, eta), self.mesh.dy_deta(xi, eta)
        txx, tyx, txy, tyy = self.func_in_form[1](x, y), self.func_in_form[0](x, y), self.func_in_form[3](x, y), self.func_in_form[2](x, y)
        cochain_local_dxodx = np.einsum('ijk,j,i->ik', (txx*J21*J11 + tyx*J11*J11 + txy*J21*J21 + tyy*J11*J21)[:self.num_basis_dxodx], quad_weights, g_eg_length.repeat(self.p+1))
        cochain_local_dyodx = np.einsum('ijk,j,i->ik', (txx*J22*J11 + tyx*J12*J11 + txy*J22*J21 + tyy*J12*J21)[self.num_basis_dxodx:self.num_basis_dxodx +self.num_basis_dyodx], quad_weights, np.tile(g_gl_length,self.p+2))
        cochain_local_dxody = np.einsum('ijk,j,i->ik', (txx*J21*J12 + tyx*J11*J12 + txy*J21*J22 + tyy*J11*J22)[self.num_basis_dxodx+self.num_basis_dyodx:-self.num_basis_dyody], quad_weights, np.repeat(g_gl_length,self.p+2))
        cochain_local_dyody = np.einsum('ijk,j,i->ik', (txx*J22*J12 + tyx*J12*J12 + txy*J22*J22 + tyy*J12*J22)[-self.num_basis_dyody:], quad_weights, np.tile(g_eg_length,self.p+1))
        print(np.shape(tyx))
        self.cochain_local = np.vstack( ( cochain_local_dxodx, cochain_local_dyodx, cochain_local_dxody, cochain_local_dyody ) )
    
    # %% projection from lobatto 0-form
    def projection_of_lobatto_0form(self, lobatto_0form, quad_type= None, quad_order=None ) :
        assert lobatto_0form.__class__.__name__ == 'Lobatto0form', 'please feed me a Lobatto0form'
        
        self.func_dxodx = self.func_dyodx = self.func_dxody = self.func_dyody = None
        self.func_dxodx_in_form = self.func_dyodx_in_form = self.func_dxody_in_form = self.func_dyody_in_form = None
    
        if quad_type is None:
            quad_type, quad_order = 'gauss', self.p+1
        
        quad_nodes, quad_weights = getattr(functionals, quad_type + '_quad')(quad_order)
        
        eg_edge_length = self._ext_gauss_nodes[1:] - self._ext_gauss_nodes[:-1]
        gl_edge_length = self._lobatto_nodes  [1:] - self._lobatto_nodes  [:-1]
        
        g_eg_length = 0.5 * eg_edge_length
        g_gl_length = 0.5 * gl_edge_length
        
        eg_start_point = self._ext_gauss_nodes[:-1].repeat(self.p+1,axis=0)
        xi_dxodx = eg_start_point[:,np.newaxis].repeat(np.size(quad_nodes),axis=1) + np.einsum('ij,i->ij',(quad_nodes[np.newaxis,:].repeat(self.num_basis_dxodx, axis=0) + 1)/2, eg_edge_length.repeat(self.p+1))
        eta_dxodx= np.tile(self._lobatto_nodes,self.p+1)[:,np.newaxis].repeat(np.size(quad_nodes),axis=1)
        
        gl_start_point = np.tile(self._lobatto_nodes[:-1],self.p+2)
        xi_dyodx = self._ext_gauss_nodes.repeat(self.p)[:,np.newaxis].repeat(np.size(quad_nodes), axis = 1)
        eta_dyodx= gl_start_point[:,np.newaxis].repeat(np.size(quad_nodes),axis=1) + np.einsum('ij,i->ij',(quad_nodes[np.newaxis,:].repeat(self.num_basis_dyodx, axis=0) + 1)/2, np.tile(gl_edge_length,self.p+2))
        
        gl_start_point = np.repeat(self._lobatto_nodes[:-1],self.p+2)
        xi_dxody = gl_start_point[:,np.newaxis].repeat(np.size(quad_nodes),axis=1) + np.einsum('ij,i->ij',(quad_nodes[np.newaxis,:].repeat(self.num_basis_dxody, axis=0) + 1)/2, gl_edge_length.repeat(self.p+2))
        eta_dxody= np.tile(self._ext_gauss_nodes,self.p)[:,np.newaxis].repeat(np.size(quad_nodes),axis=1)
        
        eg_start_point = np.tile(self._ext_gauss_nodes[:-1],self.p+1)
        xi_dyody = self._lobatto_nodes.repeat(self.p+1)[:,np.newaxis].repeat(np.size(quad_nodes), axis = 1)
        eta_dyody= eg_start_point[:,np.newaxis].repeat(np.size(quad_nodes),axis=1) + np.einsum('ij,i->ij',(quad_nodes[np.newaxis,:].repeat(self.num_basis_dyody, axis=0) + 1)/2, np.tile(eg_edge_length,self.p+1) )
        
        xi = np.vstack( (  xi_dxodx,  xi_dyodx,  xi_dxody,  xi_dyody ) )
        eta= np.vstack( ( eta_dxodx, eta_dyodx, eta_dxody, eta_dyody ) )
        
        x, y = self.mesh.mapping(xi ,eta)
#        
#        eg_edge_length = self._ext_gauss_nodes[1:] - self._ext_gauss_nodes[:-1]
#        g_eg_length = 0.5 * eg_edge_length
#        eg_start_point = self._ext_gauss_nodes[:-1]
#        
#        xi  = eg_start_point.repeat(np.size(quad_nodes)) + np.kron(eg_edge_length/2, quad_nodes+1)
#        eta = self._lobatto_nodes
#        lobatto_0form.reconstruct( xi, eta , do_plot = False)
#        
#        tyx = lobatto_0form.reconstructed.reshape((self.p+1, np.size(quad_nodes)*(self.p+1), self.mesh.num_elements), order='F').reshape( (self.p+1, np.size(quad_nodes), self.p+1, self.mesh.num_elements),order = 'F' ).transpose([0,2,1,3]).reshape( ( (self.p+1)**2, np.size(quad_nodes), self.mesh.num_elements ), order = 'F' )
#        
#        xi = self._lobatto_nodes
#        eta  = eg_start_point.repeat(np.size(quad_nodes)) + np.kron(eg_edge_length/2, quad_nodes+1)
#        lobatto_0form.reconstruct( xi, eta , do_plot = False)
#        
#        txy = lobatto_0form.reconstructed.reshape( (np.size(quad_nodes), (self.p+1)**2, self.mesh.num_elements), order = 'F' ).transpose([1,0,2])
#       
                
        return xi , eta
    # %% 
    def reconstruct(self, xi=None, eta=None, do_plot = True, do_return = False):
        assert self.cochain is not None, "no cochain to reconstruct"
        if xi is None and eta is None:
            xi  = eta = np.linspace( -1, 1, np.int( np.ceil(np.sqrt(10000 / self.mesh.num_elements)) +1 ) )
            
        self.evaluate_basis(domain=(xi, eta))
        xi, eta = self.xi.ravel('F'), self.eta.ravel('F')
        
        tyx = np.tensordot(self.basis_dxodx, self.cochain_local_dxodx, axes=((0), (0)))
        txx = np.tensordot(self.basis_dyodx, self.cochain_local_dyodx, axes=((0), (0)))
        tyy = np.tensordot(self.basis_dxody, self.cochain_local_dxody, axes=((0), (0)))
        txy = np.tensordot(self.basis_dyody, self.cochain_local_dyody, axes=((0), (0)))
        
        self.reconstructed_dxodx = txx * self.mesh.invJ21(xi,eta) * self.mesh.invJ11(xi,eta) + \
                                   tyx * self.mesh.invJ11(xi,eta) * self.mesh.invJ11(xi,eta) + \
                                   txy * self.mesh.invJ21(xi,eta) * self.mesh.invJ21(xi,eta) + \
                                   tyy * self.mesh.invJ11(xi,eta) * self.mesh.invJ21(xi,eta)
                                   
        self.reconstructed_dyodx = txx * self.mesh.invJ22(xi,eta) * self.mesh.invJ11(xi,eta) + \
                                   tyx * self.mesh.invJ12(xi,eta) * self.mesh.invJ11(xi,eta) + \
                                   txy * self.mesh.invJ22(xi,eta) * self.mesh.invJ21(xi,eta) + \
                                   tyy * self.mesh.invJ12(xi,eta) * self.mesh.invJ21(xi,eta)
                                   
        self.reconstructed_dxody = txx * self.mesh.invJ21(xi,eta) * self.mesh.invJ12(xi,eta) + \
                                   tyx * self.mesh.invJ11(xi,eta) * self.mesh.invJ12(xi,eta) + \
                                   txy * self.mesh.invJ21(xi,eta) * self.mesh.invJ22(xi,eta) + \
                                   tyy * self.mesh.invJ11(xi,eta) * self.mesh.invJ22(xi,eta)
                                   
        self.reconstructed_dyody = txx * self.mesh.invJ22(xi,eta) * self.mesh.invJ12(xi,eta) + \
                                   tyx * self.mesh.invJ12(xi,eta) * self.mesh.invJ12(xi,eta) + \
                                   txy * self.mesh.invJ22(xi,eta) * self.mesh.invJ22(xi,eta) + \
                                   tyy * self.mesh.invJ12(xi,eta) * self.mesh.invJ22(xi,eta)
        
        x, y = self.mesh.mapping(self.xi, self.eta)
        
        if do_return is True:
            return (x, y), self.reconstructed_dxodx, self.reconstructed_dyodx, self.reconstructed_dxody, self.reconstructed_dyody
        
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
    
                recon_4d_dx = self.reconstructed_dxodx.reshape(
                    num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
                reconstructed_dxodx = np.moveaxis(recon_4d_dx, 2, 1).ravel('F').reshape(
                    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')
    
                recon_4d_dx = self.reconstructed_dyodx.reshape(
                    num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
                reconstructed_dyodx = np.moveaxis(recon_4d_dx, 2, 1).ravel('F').reshape(
                    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')
                
                recon_4d_dx = self.reconstructed_dxody.reshape(
                    num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
                reconstructed_dxody = np.moveaxis(recon_4d_dx, 2, 1).ravel('F').reshape(
                    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')
                
                recon_4d_dx = self.reconstructed_dyody.reshape(
                    num_pts_y, num_pts_x, num_el_y, num_el_x, order='F')
                reconstructed_dyody = np.moveaxis(recon_4d_dx, 2, 1).ravel('F').reshape(
                    num_el_y * num_pts_y, num_el_x * num_pts_x, order='F')
            
                plt.contourf(x, y, -reconstructed_dxodx)
                plt.title("STG1 co1-form dxodx")
                plt.colorbar()
                plt.show()
                
                plt.contourf(x, y, reconstructed_dyodx)
                plt.title("STG1 co1-form dyodx")
                plt.colorbar()
                plt.show()
                
                plt.contourf(x, y, reconstructed_dxody)
                plt.title("STG1 co1-form dxody")
                plt.colorbar()
                plt.show()
                
                plt.contourf(x, y, -reconstructed_dyody)
                plt.title("STG1 co1-form dyody")
                plt.colorbar()
                plt.show()

            
    # %%    
    def L2_error(self, func = None):
        if func is not None:
            self.func = func
        assert self.func != (None, None, None, None), "exact func do not exist, no L2_error"

# %%
if __name__ == '__main__':
    crazy_mesh = CrazyMesh(2, (2, 2), ((-1, 1), (-1, 1)), curvature=0.0)
    f1 = STG1co1form(crazy_mesh,2)
    f0 = Lobatto0form(crazy_mesh,2)
    
    def p(x,y): return np.cos(np.pi*x) * np.cos(np.pi*y)
    f0.discretize(p)
    f0.reconstruct()
    a , b = f1.projection_of_lobatto_0form(f0)
#    f2 = STG1co1form(crazy_mesh,3)
    
#    f1.inner(f2)
    
#    def t_dxodx(x,y): return np.cos(np.pi*x) * np.sin(np.pi*y)
#    def t_dyodx(x,y): return np.sin(np.pi*x) * np.cos(np.pi*y)
#    
#    def t_dxody(x,y): return np.cos(np.pi*x) * np.sin(np.pi*y)
#    def t_dyody(x,y): return np.sin(np.pi*x) * np.cos(np.pi*y)
#    
#    f1.discretize((t_dxodx, t_dyodx, t_dxody, t_dyody))
#    
##    M = f1.inner(f1) [0]
#    
#    f1.reconstruct()
#    
#    E21,f2 = f1.coboundary_assembled
#    f2.reconstruct()
    