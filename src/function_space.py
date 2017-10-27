"""This module contain classes defining the Function Spaces."""
from dof_map import DofMap
from mesh import CrazyMesh
import numpy as np


class FunctionSpace(object):
    """Define a functional space."""

    def __init__(self, mesh, form_type, degree, is_inner=True, separated_dof=True):
        self.mesh = mesh
        self.form_type = form_type
        self.k = int(form_type.split('-')[0])

        if isinstance(degree, int):
            self.p = (degree, degree)
        elif isinstance(degree, tuple):
            self.p = degree

        self.is_inner = is_inner

        self.str_to_elem = {'0-lobatto': 'LobattoNodal',
                            '0-gauss': 'GaussNodal',
                            '0-ext_gauss': 'ExtGaussNodal',
                            '0': 'FlexibleNodal',
                            '1-lobatto': 'LobattoEdge',
                            '1-gauss': 'GaussEdge',
                            '1-ext_gauss': 'ExtGaussEdge',
                            '1-total_ext_gauss': 'TotalExtGaussEdge',
                            '2-lobatto': 'LobattoFace',
                            '2-gauss': 'GaussFace',
                            '2-ext_gauss': 'ExtGaussFace',
                            '2': 'FlexibleFace'}

        self.str_to_form = {'0-lobatto': 'Form_0',
                            '0-gauss': 'Form_0',
                            '0-ext_gauss': 'ExtGaussForm_0',
                            '1-lobatto': 'Form_1',
                            '1-gauss': 'Form_1',
                            '1-ext_gauss': 'ExtGaussForm_1',
                            '1-total_ext_gauss': 'Form_1',
                            '2-lobatto': 'Form_2',
                            '2-gauss': 'Form_2',
                            '2-ext_gauss': 'ExtGaussForm_2'}
        self.separated_dof = separated_dof
        self._dof_map = None
        self._num_dof = None
        self._num_internal_dof = None
        self._num_internal_local_dof = None
        self._num_local_dof = None

    @property
    def dof_map(self):
        """Return a map object between the local and the global dofs."""
        if self._dof_map is None:
            self._dof_map = DofMap(self)
        return self._dof_map

    @property
    def num_dof(self):
        """Return the number of degrees of freedom."""
        if self._num_dof is None:
            self._num_dof = np.max(self.dof_map.dof_map) + 1
        return self._num_dof

    @property
    def num_internal_dof(self):
        """Return the number of degrees of freedom."""
        if self._num_internal_dof is None:
            self._num_internal_dof = np.max(self.dof_map.dof_map_internal) + 1
        return self._num_internal_dof

    @property
    def num_internal_local_dof(self):
        """Return the number of degrees of freedom."""
        if self._num_internal_local_dof is None:
            self._num_internal_local_dof = np.size(self.dof_map.dof_map_internal[0, :])
        return self._num_internal_local_dof

    @property
    def num_local_dof(self):
        """Return the number of degrees of freedom."""
        if self._num_local_dof is None:
            self._num_local_dof = np.size(self.dof_map.dof_map[0, :])
        return self._num_local_dof


def NextSpace(func_space):
    try:
        assert func_space.k < 2
        k = func_space.k + 1
        next_form_type = str(k) + '-' + func_space.form_type.split('-')[1]
        next_func_space = FunctionSpace(func_space.mesh, next_form_type, func_space.p)
        return next_func_space
    except AssertionError as e:
        raise AssertionError("In 2D space next space is L_2")


def DualSpace(func_space, extend=True):
    """Generate the dual space to a given function space.

    Parameters
    ----------
    func_space : FunctionSpace
            User defined function space.

    Return
    ------
    dual_space : FunctionSpace
            The dual function space to func_space.

    """
    # dict mapping primal to dual spaces
    primal_to_dual = {'0-lobatto': '2-gauss', '2-gauss': '0-lobatto',
                      '0-gauss': '2-lobatto', '2-lobatto': '0-gauss',
                      '1-lobatto': '1-gauss', '1-gauss': '1-lobatto'}
    primal_to_dual_ext = {'0-lobatto': '2-ext_gauss', '2-ext_gauss': '0-lobatto',
                          '0-ext_gauss': '2-lobatto', '2-lobatto': '0-ext_gauss',
                          '1-lobatto': '1-ext_gauss', '1-ext_gauss': '1-lobatto'}
    # find the dual element family
    if extend:
        dual_form_type = primal_to_dual_ext[func_space.form_type]
    else:
        dual_form_type = primal_to_dual[func_space.form_type]

    # redifine the p values for the dual space
    if 'gauss' in dual_form_type:
        p_dual = (func_space.p[0] - 1, func_space.p[1] - 1)
    elif 'lobatto' in dual_form_type:
        p_dual = (func_space.p[0] + 1, func_space.p[1] + 1)
    # switch the orientation
    dual_is_inner = not func_space.is_inner
    dual_space = FunctionSpace(func_space.mesh, dual_form_type, p_dual, dual_is_inner)
    return dual_space

if __name__ == '__main__':
    mesh = CrazyMesh(2, (1, 1), ((-1, 1), (-1, 1)))
    func_space = FunctionSpace(mesh, '0-lobatto', (2, 2))
