"""Test for basis forms."""
import path_magic
import unittest
from mesh import CrazyMesh
from basis_forms import BasisForm
from hodge import hodge
from function_space import FunctionSpace, DualSpace, NextSpace
from forms import Form
import numpy as np
from coboundaries import d
import numpy.testing as npt
from inner_product import inner
from assemble import assemble


class TestAssemble(unittest.TestCase):

    def setUp(self):
        self.mesh = CrazyMesh(2, (4, 5), ((-1., 1), (-1, 1)))
        self.p = (3, 3)
        # print(self.func_space.dof_map.dof_map)

    # @unittest.skip
    def test_assembly_inner_product_0_forms(self):
        """Test the assembly of 1-forms inner products."""
        func_space_lob_0 = FunctionSpace(self.mesh, '0-lobatto', self.p)
        func_space_gauss_0 = FunctionSpace(self.mesh, '0-gauss', self.p)
        func_space_extgauss_0 = FunctionSpace(self.mesh, '0-ext_gauss', self.p)

        basis_lob = BasisForm(func_space_lob_0)
        basis_lob.quad_grid = 'gauss'
        M_0_lob = inner(basis_lob, basis_lob)

        basis_gauss = BasisForm(func_space_gauss_0)
        basis_gauss.quad_grid = 'lobatto'
        M_0_gauss = inner(basis_gauss, basis_gauss)

        basis_ext_gauss = BasisForm(func_space_extgauss_0)
        basis_ext_gauss.quad_grid = 'lobatto'
        M_0_extgauss = inner(basis_ext_gauss, basis_ext_gauss)

        M_0_lob_ass_ref = assemble_slow(self.mesh, M_0_lob, func_space_lob_0.dof_map.dof_map,
                                        func_space_lob_0.dof_map.dof_map)
        M_0_gauss_ass_ref = assemble_slow(self.mesh, M_0_gauss, func_space_gauss_0.dof_map.dof_map,
                                          func_space_gauss_0.dof_map.dof_map)
        M_0_extgauss_ass_ref = assemble_slow(
            self.mesh, M_0_extgauss, func_space_extgauss_0.dof_map.dof_map_internal, func_space_extgauss_0.dof_map.dof_map_internal)

        M_0_lob_ass = assemble(M_0_lob, func_space_lob_0, func_space_lob_0).toarray()
        M_0_gauss_ass = assemble(M_0_gauss, func_space_gauss_0, func_space_gauss_0).toarray()
        M_0_extgauss_ass = assemble(M_0_extgauss, func_space_extgauss_0,
                                    func_space_extgauss_0).toarray()

        npt.assert_array_almost_equal(M_0_lob_ass_ref, M_0_lob_ass)
        npt.assert_array_almost_equal(M_0_gauss_ass_ref, M_0_gauss_ass)
        npt.assert_array_almost_equal(M_0_extgauss_ass_ref, M_0_extgauss_ass)

    # @unittest.skip
    def test_assembly_inner_product_1_forms(self):
        """Test the assembly of 1-forms inner products."""
        func_space_lob = FunctionSpace(self.mesh, '1-lobatto', self.p)
        func_space_gauss = FunctionSpace(self.mesh, '1-gauss', self.p)
        func_space_extgauss = FunctionSpace(self.mesh, '1-ext_gauss', self.p)

        basis_lob = BasisForm(func_space_lob)
        basis_lob.quad_grid = 'gauss'
        M_lob = inner(basis_lob, basis_lob)

        basis_gauss = BasisForm(func_space_gauss)
        basis_gauss.quad_grid = 'lobatto'
        M_gauss = inner(basis_gauss, basis_gauss)

        basis_ext_gauss = BasisForm(func_space_extgauss)
        basis_ext_gauss.quad_grid = 'lobatto'
        M_extgauss = inner(basis_ext_gauss, basis_ext_gauss)

        M_lob_ass_ref = assemble_slow(self.mesh, M_lob, func_space_lob.dof_map.dof_map,
                                      func_space_lob.dof_map.dof_map)
        M_gauss_ass_ref = assemble_slow(self.mesh, M_gauss, func_space_gauss.dof_map.dof_map,
                                        func_space_gauss.dof_map.dof_map)
        M_extgauss_ass_ref = assemble_slow(
            self.mesh, M_extgauss, func_space_extgauss.dof_map.dof_map_internal, func_space_extgauss.dof_map.dof_map_internal)

        M_lob_ass = assemble(M_lob, func_space_lob, func_space_lob).toarray()
        M_gauss_ass = assemble(M_gauss, func_space_gauss, func_space_gauss).toarray()
        M_extgauss_ass = assemble(M_extgauss, func_space_extgauss,
                                  func_space_extgauss).toarray()

        npt.assert_array_almost_equal(M_lob_ass_ref, M_lob_ass)
        npt.assert_array_almost_equal(M_gauss_ass_ref, M_gauss_ass)
        npt.assert_array_almost_equal(M_extgauss_ass_ref, M_extgauss_ass)

    # @unittest.skip
    def test_assembly_inner_product_2_forms(self):
        """Test the assembly of 1-forms inner products."""
        func_space_lob = FunctionSpace(self.mesh, '2-lobatto', self.p)
        func_space_gauss = FunctionSpace(self.mesh, '2-gauss', self.p)
        func_space_extgauss = FunctionSpace(self.mesh, '2-ext_gauss', self.p)

        basis_lob = BasisForm(func_space_lob)
        basis_lob.quad_grid = 'gauss'
        M_lob = inner(basis_lob, basis_lob)

        basis_gauss = BasisForm(func_space_gauss)
        basis_gauss.quad_grid = 'lobatto'
        M_gauss = inner(basis_gauss, basis_gauss)

        basis_ext_gauss = BasisForm(func_space_extgauss)
        print(basis_ext_gauss.num_basis)
        basis_ext_gauss.quad_grid = 'lobatto'
        M_extgauss = inner(basis_ext_gauss, basis_ext_gauss)

        M_lob_ass_ref = assemble_slow(self.mesh, M_lob, func_space_lob.dof_map.dof_map,
                                      func_space_lob.dof_map.dof_map)
        M_gauss_ass_ref = assemble_slow(self.mesh, M_gauss, func_space_gauss.dof_map.dof_map,
                                        func_space_gauss.dof_map.dof_map)
        M_extgauss_ass_ref = assemble_slow(
            self.mesh, M_extgauss, func_space_extgauss.dof_map.dof_map_internal, func_space_extgauss.dof_map.dof_map_internal)

        M_lob_ass = assemble(M_lob, func_space_lob, func_space_lob).toarray()
        M_gauss_ass = assemble(M_gauss, func_space_gauss, func_space_gauss).toarray()
        M_extgauss_ass = assemble(M_extgauss, func_space_extgauss,
                                  func_space_extgauss).toarray()

        npt.assert_array_almost_equal(M_lob_ass_ref, M_lob_ass)
        npt.assert_array_almost_equal(M_gauss_ass_ref, M_gauss_ass)
        npt.assert_array_almost_equal(M_extgauss_ass_ref, M_extgauss_ass)

    # @unittest.skip
    def test_assemble_hodge(self):
        p_dual = (self.p[0] - 1, self.p[1] - 1)
        func_space_lob_0 = FunctionSpace(self.mesh, '0-lobatto', self.p)
        func_space_extgauss_0 = FunctionSpace(self.mesh, '0-ext_gauss', p_dual)

        func_space_lob_2 = FunctionSpace(self.mesh, '2-lobatto', self.p)
        func_space_extgauss_2 = FunctionSpace(self.mesh, '2-ext_gauss', p_dual)

        func_space_lob_1 = FunctionSpace(self.mesh, '1-lobatto', self.p)
        func_space_extgauss_1 = FunctionSpace(self.mesh, '1-ext_gauss', p_dual)

        hodge_20_ext = hodge(func_space_extgauss_0)

        hodge_11_lob = hodge(func_space_lob_1)

        hodge_02_lob = hodge(func_space_lob_2)

        hodge_assembled_ref = assemble_slow(
            self.mesh, hodge_20_ext, func_space_lob_2.dof_map.dof_map, func_space_extgauss_0.dof_map.dof_map_internal)
        hodge_assembled = assemble(
            hodge_20_ext, (func_space_lob_2, func_space_extgauss_0)).toarray()

        npt.assert_array_almost_equal(hodge_assembled_ref, hodge_assembled)

        hodge_assembled_ref = assemble_slow(
            self.mesh, hodge_11_lob, func_space_extgauss_1.dof_map.dof_map_internal, func_space_lob_1.dof_map.dof_map)
        hodge_assembled = assemble(
            hodge_11_lob, (func_space_extgauss_1, func_space_lob_1)).toarray()

        npt.assert_array_almost_equal(hodge_assembled_ref, hodge_assembled)
        hodge_assembled_ref = assemble_slow(
            self.mesh, hodge_02_lob, func_space_extgauss_0.dof_map.dof_map_internal, func_space_lob_2.dof_map.dof_map)
        hodge_assembled = assemble(
            hodge_02_lob, (func_space_extgauss_0, func_space_lob_2)).toarray()

        npt.assert_array_almost_equal(hodge_assembled_ref, hodge_assembled)

    # @unittest.skip
    def test_assemble_incidence_matrices(self):
        p_dual = (self.p[0] - 1, self.p[1] - 1)
        func_space_lob_0 = FunctionSpace(self.mesh, '0-lobatto', self.p)
        func_space_extgauss_0 = FunctionSpace(self.mesh, '0-ext_gauss', p_dual)

        func_space_lob_1 = NextSpace(func_space_lob_0)
        func_space_extgauss_1 = FunctionSpace(self.mesh, '1-ext_gauss', p_dual)

        func_space_lob_2 = FunctionSpace(self.mesh, '2-lobatto', self.p)
        func_space_extgauss_2 = FunctionSpace(self.mesh, '2-ext_gauss', p_dual)

        e10_lob = d(func_space_lob_0)
        e21_lob = d(func_space_lob_1)
        e10_ext = d(func_space_extgauss_0)
        # e21_ext = d(func_space_extgauss_1)
        #
        e10_lob_assembled_ref = assemble_slow(
            self.mesh, e10_lob, func_space_lob_1.dof_map.dof_map, func_space_lob_0.dof_map.dof_map, mode='replace')

        e21_lob_assembled_ref = assemble_slow(
            self.mesh, e21_lob, func_space_lob_2.dof_map.dof_map,
            func_space_lob_1.dof_map.dof_map, mode='replace')

        e10_ext_assembled_ref = assemble_slow(
            self.mesh, e10_ext, func_space_extgauss_1.dof_map.dof_map,
            func_space_extgauss_0.dof_map.dof_map, mode='replace')

        # e21_ext_assembled_ref = assemble_slow(
        #     self.mesh, e21_ext, func_space_extgauss_2.dof_map.dof_map,
        #     func_space_extgauss_1.dof_map.dof_map, mode='replace')

        e10_lob_assembled = assemble(e10_lob, (func_space_lob_1, func_space_lob_0)).toarray()
        e21_lob_assembled = assemble(e21_lob, (func_space_lob_2, func_space_lob_1)).toarray()
        e10_ext_assembled = assemble(e10_ext, (func_space_extgauss_1,
                                               func_space_extgauss_0)).toarray()
        # e21_ext_assembled = assemble(e21_ext, func_space_extgauss_2,
        #                              func_space_extgauss_1).toarray()

        npt.assert_array_almost_equal(e10_lob_assembled_ref, e10_lob_assembled)
        npt.assert_array_almost_equal(e21_lob_assembled_ref, e21_lob_assembled)
        npt.assert_array_almost_equal(e10_ext_assembled_ref, e10_ext_assembled)
        # npt.assert_array_almost_equal(e21_ext_assembled_ref, e21_ext_assembled)

        e10_internal = d(func_space_extgauss_0)[:func_space_extgauss_1.num_internal_local_dof]

        e10_internal_assembled_ref = assemble_slow(
            self.mesh, e10_internal, func_space_extgauss_1.dof_map.dof_map_internal, func_space_extgauss_0.dof_map.dof_map)
        e10_internal_assembled = assemble(
            e10_internal, (func_space_extgauss_1, func_space_extgauss_0)).toarray()

        npt.assert_array_almost_equal(e10_internal_assembled_ref, e10_internal_assembled)


def assemble_slow(mesh, M2bass, Gi_, G_j, mode='add', assembler='baseV00', info=None):
    """
    The caller

    #INPUTS----------
        #ESSENTIAL:
            M2bass :: (2-d or 3-d matrix) the matrix to be assembled.
            Gi_ :: (matrix) the gathering matrix corresponding to M2bass(:,).
            G_j :: (matrix) the gathering matrix corresponding to M2bass(,:).
            mesh :: (class) the mesh
        #OPTIONAL:
            mode :: (strings, default: 'add') the mode of doing assembling.
                            'add' or 'replace' or 'avarage'
        assembler :: (strings, default: 'BaseV00') which assembler we use.
        info :: the other information we may need
    #OUTPUTS---------
             Assembled :: (matrix) the assembled matrix.
    #EXAMPLES--------
    #NOTES-----------
    """
    print(">>> do assembling @mode=", mode, "| @assembler=", assembler)
# data check
    sz = np.shape(M2bass)
    szsz = np.shape(sz)
    if szsz[0] == 2:  # M2bass is a matrix, so we assume for each element, the matrix is the same
        print("      ~ sub-matrices are same......")
        M2bass = np.repeat(M2bass[:, :, np.newaxis], mesh.num_elements, axis=2)
    elif szsz[0] == 3:
        if sz[2] != mesh.num_elements:
            raise Exception("data structure wrong......")
        else:
            pass
    else:
        raise Exception("data structure wrong......")

    # print("      > data check......")
    szGi_ = np.shape(Gi_)
    szG_j = np.shape(G_j)
    rowGi_ = szGi_[1]
    columnGi_ = szGi_[0]
    rowG_j = szG_j[1]
    columnG_j = szG_j[0]

    sz = np.shape(M2bass[:, :, 0])
    row = sz[0]
    colume = sz[1]
    if (row == rowGi_) and (colume == rowG_j) and (columnGi_ == columnG_j) and (columnG_j == mesh.num_elements):
        pass
    else:
        raise Exception("data structure wrong......")
    # print("      < data check pass")

# call the assembler and do the assembling
    if assembler == 'baseV00':
        assembled = _assembler_baseV00(M2bass, Gi_, G_j, mode)
    else:
        raise Exception("assembler not coded yet......")
    return assembled


def _assembler_baseV00(M2bass, Gi_, G_j, mode):
    """
    The mose basic assembler, and of course, the slowest one

    """
    Gi_ = Gi_.T
    G_j = G_j.T

    hmgeoiti_ = int(np.max(Gi_) + 1)
    hmgeoit_j = int(np.max(G_j) + 1)

    szGi_ = np.shape(Gi_)
    szG_j = np.shape(G_j)
    rowGi_ = szGi_[0]
    rowG_j = szG_j[0]
    num_elements = szG_j[1]

    # assembled = lil_matrix((hmgeoiti_, hmgeoit_j))
    assembled = np.zeros(shape=(hmgeoiti_, hmgeoit_j), order='F')

    if mode == 'add':
        for k in range(num_elements):
            E = M2bass[:, :, k]
            for a in range(rowGi_):
                i = int(Gi_[a, k])
                for b in range(rowG_j):
                    j = int(G_j[b, k])
                    assembled[i, j] = assembled[i, j] + E[a, b]

    elif mode == 'replace':
        for k in range(num_elements):
            E = M2bass[:, :, k]
            for a in range(rowGi_):
                i = int(Gi_[a, k])
                for b in range(rowG_j):
                    j = int(G_j[b, k])
                    assembled[i, j] = E[a, b]

    elif mode == 'average':
        asstimes = np.zeros((hmgeoiti_, 1))
        for k in range(num_elements):
            E = M2bass[:, :, k]
            for a in range(rowGi_):
                i = int(Gi_[a, k])
                asstimes[i] = asstimes[i] + 1
                for b in range(rowG_j):
                    j = int(G_j[b, k])
                    assembled[i, j] = assembled[i, j] + E[a, b]

        for i in range(hmgeoiti_):
            if asstimes[i] > 1:
                assembled[i, :] = assembled[i, :] / asstimes[i]

    else:
        raise Exception('Mode wrong: add, replace or average......')

    return assembled


if __name__ == '__main__':
    unittest.main()
