from scipy import sparse
from tqdm import *
import numpy as np
from scipy.sparse import coo_matrix

# %% --------------------------------------------------------------------------
def assemble_(M2bass, Gi_, G_j, mode='add', assembler='advanced', info=None):
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
    num_elements = np.shape(Gi_)[0]
    if M2bass.ndim == 2:
        M2bass = np.repeat((M2bass)[:, :, np.newaxis], num_elements, axis=2)

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
    if (row == rowGi_) and (colume == rowG_j) and (columnGi_ == columnG_j) and (columnG_j == num_elements):
        pass
    else:
        raise Exception("data structure wrong......")
    # print("      < data check pass")

# call the assembler and do the assembling
    if assembler == 'baseV00':
        print(">>> do assembling @mode=", mode, "| @assembler=", assembler)
        assembled = _assembler_baseV00(M2bass, Gi_, G_j, mode)
    elif assembler == 'advanced':
        assembled = assemble_array(M2bass, Gi_, G_j, assemble=True, order='F')
    else:
        raise Exception("assembler not coded yet......")
    return assembled

# %% --------------------------------------------------------------------------
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

    assembled = sparse.lil_matrix((hmgeoiti_, hmgeoit_j))
#    assembled = np.zeros(shape=(hmgeoiti_, hmgeoit_j), order='F')

    if mode == 'add':
        for k in tqdm(range(num_elements)):
            E = M2bass[:, :, k]
            for a in range(rowGi_):
                i = int(Gi_[a, k])
                for b in range(rowG_j):
                    j = int(G_j[b, k])
                    assembled[i, j] = assembled[i, j] + E[a, b]

    elif mode == 'replace':
        for k in tqdm(range(num_elements)):
            E = M2bass[:, :, k]
            for a in range(rowGi_):
                i = int(Gi_[a, k])
                for b in range(rowG_j):
                    j = int(G_j[b, k])
                    assembled[i, j] = E[a, b]

    elif mode == 'average':
        asstimes = np.zeros((hmgeoiti_, 1))
        for k in tqdm(range(num_elements)):
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

def assemble_array(array, dof_map_rows, dof_map_cols, assemble=False, order='F'):
    """Convert the element-wise inner products to a sparse global matrix.

    Using the global numering matrix is possible to map the inner product
    of the degrees of freedom, using the local numbering,
    into the global matrix of inner products.

    Parameters
    ----------
    array_local : ndarray
        3d array containing the inner product of basis functions elementwise.
        The third dimension moves you through the elements.
    func_space : FunctionSpace
        Function space of the basis functions used to generate the inner product
    assemble : bool, optional
        Determines if the global matrix is assembled

    Returns
    -------
    inner_sparse : crs array, optional
        Sparse global inner product
    row_idx : ndarray
        array of row global indeces for the inner product
    column_idx : ndarray
        array of column global indeces for the inner product
    data : ndarray
        flattened array of containing the inner product values

    Notes
    -----
    row_idx, column_idx, data can be used to quickly build a sparse matrix.

    >>> coo_matrix((data, (row_idx, column_idx))

    """
    # hypothetical final dimensions of the matrix
    dim_x, dim_y = np.shape(array)[:-1]

    num_dof_rows = np.max(dof_map_rows) + 1
    num_dof_per_element_rows = np.size(dof_map_rows[0, :])

    num_dof_cols = np.max(dof_map_cols) + 1
    num_dof_per_element_cols = np.size(dof_map_cols[0, :])
    
    # rows and column indexes for the global inner product
    row_idx = np.tile(dof_map_rows, num_dof_per_element_cols).ravel()
    column_idx = np.repeat(dof_map_cols, num_dof_per_element_rows)
    data = array.ravel(order)

    inner_sparse = coo_matrix(
        (data, (row_idx, column_idx)), shape=(num_dof_rows, num_dof_cols))
    return inner_sparse.tocsc()