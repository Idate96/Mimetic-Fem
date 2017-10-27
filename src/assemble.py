"""This modules contains assembly routines for global matrices."""
from scipy.sparse import coo_matrix
from function_space import FunctionSpace
import numpy as np


def assemble(array, func_spaces, assemble=True):
    """General assembly procedure for local arrays."""

    if isinstance(func_spaces, FunctionSpace):
        # set row and col function spaces the same
        func_spaces = (func_spaces, func_spaces)
    # if the array is 2d (wedge product, incidence matrix ...)
    if array.ndim == 2:
        result = assemble_local_array(array, *func_spaces, assemble=assemble)
    if array.ndim == 3:
        result = assemble_array(array, *func_spaces, assemble=assemble)

    return result


def assemble_local_array(local_array, func_space_rows, func_space_cols, assemble=True):
    """Assemble a local 2d array matrix."""
    # extrude the gathering matrix in the third dimension
    data = np.repeat((local_array)[:, :, np.newaxis],
                     func_space_rows.mesh.num_elements, axis=2)
    result = assemble_array(data, func_space_rows, func_space_cols, assemble)
    return result


def assemble_array(array, func_space_rows, func_space_cols, assemble=False, order='F'):
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

    dof_map_rows, num_dof_rows = get_dof_map(dim_x, func_space_rows)
    num_dof_per_element_rows = np.size(dof_map_rows[0, :])

    dof_map_cols, num_dof_cols = get_dof_map(dim_y, func_space_cols)
    num_dof_per_element_cols = np.size(dof_map_cols[0, :])
    # rows and column indexes for the global inner product
    row_idx = np.tile(dof_map_rows, num_dof_per_element_cols).ravel()
    column_idx = np.repeat(dof_map_cols, num_dof_per_element_rows)
    data = array.ravel(order)

    if assemble:
        inner_sparse = coo_matrix(
            (data, (row_idx, column_idx)), shape=(num_dof_rows, num_dof_cols))
        return inner_sparse.tocsc()
    else:
        return row_idx, column_idx, data


def get_dof_map(dimension, function_space):
    """Return the dof map for the assembly procedure."""
    # check if the element is extended
    dof_map = None
    if 'ext_gauss' == function_space.form_type.split('-')[1]:
        # use if the necessary the interal map for the degrees of freedom
        if dimension == function_space.num_internal_local_dof:
            dof_map = function_space.dof_map.dof_map_internal
            num_dof = function_space.num_internal_dof

    if dimension == function_space.num_local_dof:
        dof_map = function_space.dof_map.dof_map
        num_dof = function_space.num_dof

    if dof_map is None:
        raise ValueError("The dimensions of the matrix ({0}) to assemble to not match the number of degrees of freedom of the element ({1})." .format(
            dimension, function_space.num_local_dof))
    return dof_map, num_dof


# def assemble_square_arrays(array_local, func_space, assemble=False):
#     """Convert the element-wise inner products to a sparse global matrix.
#
#     Using the global numering matrix is possible to map the inner product
#     of the degrees of freedom, using the local numbering,
#     into the global matrix of inner products.
#
#     Parameters
#     ----------
#     array_local : ndarray
#         3d array containing the inner product of basis functions elementwise.
#         The third dimension moves you through the elements.
#     func_space : FunctionSpace
#         Function space of the basis functions used to generate the inner product
#     assemble : bool, optional
#         Determines if the global matrix is assembled
#
#     Returns
#     -------
#     inner_sparse : crs array, optional
#         Sparse global inner product
#     row_idx : ndarray
#         array of row global indeces for the inner product
#     column_idx : ndarray
#         array of column global indeces for the inner product
#     data : ndarray
#         flattened array of containing the inner product values
#
#     Notes
#     -----
#     row_idx, column_idx, data can be used to quickly build a sparse matrix.
#
#     >>> coo_matrix((data, (row_idx, column_idx))
#
#     """
#     dof_map = func_space.dof_map.dof_map
#     num_dof_per_element = np.size(dof_map[0, :])
#     # rows and column indexes for the global inner product
#     # to each basis funcs has associated num_dof**2 values
#     row_idx = np.repeat(dof_map, num_dof_per_element)
#     column_idx = np.tile(dof_map, num_dof_per_element).ravel()
#     data = array_local.ravel('F')
#     if assemble:
#         inner_sparse = coo_matrix(
#             (data, (row_idx, column_idx)), shape=(func_space.num_dof, func_space.num_dof))
#         return inner_sparse.tocsr()
#     else:
#         return row_idx, column_idx, data


if __name__ == '__main__':
    pass
