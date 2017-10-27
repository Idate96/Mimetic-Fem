import numpy as np


def inner(state_row, state_col, *args):
    """Calculate inner product."""
    # both trial and test function have associated coboundaries
    if isinstance(state_row, tuple) and isinstance(state_col, tuple):
        form_row, d_row = state_row
        form_col, d_col = state_col
        mass_matrix = form_row.inner(form_col, *args)
        inner_prod = np.tensordot(d_col, np.tensordot(
            d_row, mass_matrix, axes=((0), (0))), axes=((0), (1)))
    # only test function has coboundary
    elif isinstance(state_row, tuple):
        form_row, d_row = state_row
        mass_matrix = form_row.inner(state_col, *args)
        inner_prod = np.tensordot(d_row, mass_matrix, axes=((0), (0)))
    # only trial function has cobondary
    elif isinstance(state_col, tuple):
        form_col, d_col = state_col
        mass_matrix = state_row.inner(form_col, *args)
        inner_prod = np.swapaxes(np.tensordot(d_col, mass_matrix, axes=((0), (1))), 0, 1)

    else:
        inner_prod = state_row.inner(state_col, *args)
    return inner_prod


class MeshFunction(object):
    """This class can be used to introduce constitutive laws into the inner product."""

    def __init__(self, mesh):
        self.mesh = mesh
        self._continuous_tensor = list()
        self._discrete_tensor = list()

        self._tensor = list()
        self._inverse = list()

    @property
    def tensor(self):
        """Return components of the tensor."""
        return self._tensor

    @tensor.setter
    def tensor(self, tensor_value):
        self._tensor = tensor_value

    @property
    def discrete_tensor(self):
        """Return the dicrete version of the tensor.

        The tensor components are defined by floats. it tensor must have a value for each of the components for all the elements.
        """
        return self._discrete_tensor

    @discrete_tensor.setter
    def discrete_tensor(self, tensor_components):
        try:
            # if 2D
            assert len(tensor_components) == 3
            for component in tensor_components:
                assert np.shape(component)[-1] == self.mesh.num_elements
                self._discrete_tensor = tensor_components
        except AssertionError:
            raise AssertionError("Tensor not defined for all the elements")
    # %%
    @property
    def continous_tensor(self):
        """Return the continous version of the tensor.

        The tensor must have a function for each of the component.
        The domain of the function is assumed to be the whole physical domain.
        """
        return self._continuous_tensor
    
    @continous_tensor.setter
    def continous_tensor(self, funcs):
        try:
            for func in funcs:
                assert callable(func)
            self._continuous_tensor = funcs
        except AssertionError:
            raise TypeError("The tensor componets must be callable")
            
    # %%
    @property
    def inverse(self):
        """Return the inverse of the tensor."""
        if not self._inverse:
            self.invert_tensor()
        return self._inverse

    @inverse.setter
    def inverse(self, inverse_tensor):
        try:
            # if 2D
            assert len(inverse_tensor) == 3
            for component in inverse_tensor:
                assert np.shape(component)[-1] == self.mesh.num_elements
            self._inverse = inverse_tensor
        except AssertionError:
            raise AssertionError("Inverse tensor not defined for all the elements")

    def eval_tensor(self, xi, eta):
        """Evaluate the tensor function in the domain."""
        # make sure xi and eta have same shape
        try:
            assert np.shape(xi) == np.shape(eta)
        except AssertionError:
            raise AssertionError("xi and eta should have the same shape")

        if self.continous_tensor:
            x, y = self.mesh.mapping(xi, eta)
            # evaluate the components of the tensor for the whole domain
            self.tensor = [tensor_func(x, y) for tensor_func in self.continous_tensor]

        elif self.discrete_tensor:
            # reshape if necessary the discrete tensor to adapt it to the domain
            self.tensor = self.discrete_tensor

    def invert_tensor(self):
        """Calculate the inverse of the tensor."""
        det_k = self.tensor[0] * self.tensor[2] - self.tensor[1]**2
        self.inverse = [tens_comp / det_k for tens_comp in reversed(self.tensor)]
        # k_12_inv = -k_12/det_k
        self.inverse[1] *= -1


def diff_tens_11(x, y):
    return 1 * np.ones(np.shape(x))
    # , np.zeros(np.shape(x)), np.zeros(np.shape(x))


def diff_tens_12(x, y):
    return 0 * np.ones(np.shape(x))
    # , np.zeros(np.shape(x)), np.zeros(np.shape(x))


def diff_tens_22(x, y):
    return 1 * np.ones(np.shape(x))
    # , np.zeros(np.shape(x)), np.zeros(np.shape(x))


if __name__ == '__main__':
    pass
