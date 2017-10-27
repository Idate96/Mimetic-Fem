from function_space import FunctionSpace, DualSpace
import numpy as np
from numpy.linalg import inv
from mesh import CrazyMesh
from forms import Form, AbstractForm
from coboundaries import d
from inner_product import inner
import matplotlib.pyplot as plt


def pfun(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def hodge_from_func_space(function_space, extend):
    """Calculate the hodge matrix  from the function space."""
    dual_space = DualSpace(function_space, extend)
    dual_form = Form(dual_space)
    form = Form(function_space)
    wedge_prod = form.basis.wedged(dual_form.basis)
    inner_prod = inner(dual_form.basis, dual_form.basis)
    inverse_inner = inv(np.rollaxis(inner_prod, 2, 0))
    hodge_matrix = np.tensordot(inverse_inner, wedge_prod, axes=((2), (0)))
    hodge_matrix = np.moveaxis(hodge_matrix, 0, -1)
    return hodge_matrix


def hodge_from_form(form, extend):
    """Calculate the hodge matrix and resulting form."""
    dual_space = DualSpace(form.function_space, extend)
    dual_form = Form(dual_space)
    wedge_prod = form.basis.wedged(dual_form.basis)
    inner_prod = inner(dual_form.basis, dual_form.basis)
    inverse_inner = inv(np.rollaxis(inner_prod, 2, 0))
    hodge_matrix = np.tensordot(inverse_inner, wedge_prod, axes=((2), (0)))
    dual_form.cochain_local = np.einsum('kij,jk->ik', hodge_matrix, form.cochain_local)
    hodge_matrix = np.moveaxis(hodge_matrix, 0, -1)
    return dual_form, hodge_matrix


def hodge(*args, return_matrix=False, extend=True):
    """Perform the projection onto the dual space."""
    if isinstance(args[0], FunctionSpace):
        hodge_matrix = hodge_from_func_space(args[0], extend)
        return hodge_matrix
    if isinstance(args[0], AbstractForm):
        dual_form, hodge_matrix = hodge_from_form(args[0], extend)
        if not return_matrix:
            return dual_form
        else:
            return dual_form, hodge_matrix
