from multiprocessing import Process
from numpy.linalg import lapack_lite
from scipy import linalg

import numpy as np

# Looking one step deeper, we see that solve performs many sanity checks.
# Stripping these, we have:


def faster_inverse(A):
    b = np.identity(A.shape[2], dtype=A.dtype)

    n_eq = A.shape[1]
    n_rhs = A.shape[2]
#    pivots = np.zeros(n_eq, np.intc)
    identity = np.eye(n_eq)

    def lapack_inverse(a):
        b = np.copy(identity)
        pivots = np.zeros(n_eq, np.intc)
        results = linalg.lapack.dgesv(n_eq, n_rhs, a, n_eq, pivots, b, n_eq, 0)
        if results['info'] > 0:
            raise LinAlgError('Singular matrix')
        return b

    return np.array([lapack_inverse(a) for a in A])


# If the active Python interpreter is *NOT* optimized (e.g., option "-O" was
# *NOT* passed to this interpreter), enable type checking.
if __debug__:
    import inspect
    from functools import wraps
    from inspect import Parameter, Signature

    def beartype(func: callable) -> callable:
        '''
        Decorate the passed **callable** (e.g., function, method) to validate
        both all annotated parameters passed to this callable _and_ the
        annotated value returned by this callable if any.

        This decorator performs rudimentary type checking based on Python 3.x
        function annotations, as officially documented by PEP 484 ("Type
        Hints"). While PEP 484 supports arbitrarily complex type composition,
        this decorator requires _all_ parameter and return value annotations to
        be either:

        * Classes (e.g., `int`, `OrderedDict`).
        * Tuples of classes (e.g., `(int, OrderedDict)`).

        If optimizations are enabled by the active Python interpreter (e.g., due
        to option `-O` passed to this interpreter), this decorator is a noop.

        Raises
        ----------
        NameError
            If any parameter has the reserved name `__beartype_func`.
        TypeError
            If either:
            * Any parameter or return value annotation is neither:
              * A type.
              * A tuple of types.
            * The kind of any parameter is unrecognized. This should _never_
              happen, assuming no significant changes to Python semantics.
        '''

        # Raw string of Python statements comprising the body of this wrapper,
        # including (in order):
        #
        # * A "@wraps" decorator propagating the name, docstring, and other
        #   identifying metadata of the original function to this wrapper.
        # * A private "__beartype_func" parameter initialized to this function.
        #   In theory, the "func" parameter passed to this decorator should be
        #   accessible as a closure-style local in this wrapper. For unknown
        #   reasons (presumably, a subtle bug in the exec() builtin), this is
        #   not the case. Instead, a closure-style local must be simulated by
        #   passing the "func" parameter to this function at function
        #   definition time as the default value of an arbitrary parameter. To
        #   ensure this default is *NOT* overwritten by a function accepting a
        #   parameter of the same name, this edge case is tested for below.
        # * Assert statements type checking parameters passed to this callable.
        # * A call to this callable.
        # * An assert statement type checking the value returned by this
        #   callable.
        #
        # While there exist numerous alternatives (e.g., appending to a list or
        # bytearray before joining the elements of that iterable into a string),
        # these alternatives are either slower (as in the case of a list, due to
        # the high up-front cost of list construction) or substantially more
        # cumbersome (as in the case of a bytearray). Since string concatenation
        # is heavily optimized by the official CPython interpreter, the simplest
        # approach is (curiously) the most ideal.
        func_body = '''
@wraps(__beartype_func)
def func_beartyped(*args, __beartype_func=__beartype_func, **kwargs):
'''

        # "inspect.Signature" instance encapsulating this callable's signature.
        func_sig = inspect.signature(func)

        # Human-readable name of this function for use in exceptions.
        func_name = func.__name__ + '()'

        # For the name of each parameter passed to this callable and the
        # "inspect.Parameter" instance encapsulating this parameter (in the
        # passed order)...
        for func_arg_index, func_arg in enumerate(func_sig.parameters.values()):
            # If this callable redefines a parameter initialized to a default
            # value by this wrapper, raise an exception. Permitting this
            # unlikely edge case would permit unsuspecting users to
            # "accidentally" override these defaults.
            if func_arg.name == '__beartype_func':
                raise NameError(
                    'Parameter {} reserved for use by @beartype.'.format(
                        func_arg.name))

            # If this parameter is both annotated and non-ignorable for purposes
            # of type checking, type check this parameter.
            if (func_arg.annotation is not Parameter.empty and
                    func_arg.kind not in _PARAMETER_KIND_IGNORED):
                # Validate this annotation.
                _check_type_annotation(
                    annotation=func_arg.annotation,
                    label='{} parameter {} type'.format(
                        func_name, func_arg.name))

                # String evaluating to this parameter's annotated type.
                func_arg_type_expr = (
                    '__beartype_func.__annotations__[{!r}]'.format(
                        func_arg.name))

                # String evaluating to this parameter's current value when
                # passed as a keyword.
                func_arg_value_key_expr = 'kwargs[{!r}]'.format(func_arg.name)

                # If this parameter is keyword-only, type check this parameter
                # only by lookup in the variadic "**kwargs" dictionary.
                if func_arg.kind is Parameter.KEYWORD_ONLY:
                    func_body += '''
    if {arg_name!r} in kwargs and not isinstance(
        {arg_value_key_expr}, {arg_type_expr}):
        raise TypeError(
            '{func_name} keyword-only parameter '
            '{arg_name}={{}} not a {{!r}}'.format(
                {arg_value_key_expr}, {arg_type_expr}))
'''.format(
                        func_name=func_name,
                        arg_name=func_arg.name,
                        arg_type_expr=func_arg_type_expr,
                        arg_value_key_expr=func_arg_value_key_expr,
                    )
                # Else, this parameter may be passed either positionally or as
                # a keyword. Type check this parameter both by lookup in the
                # variadic "**kwargs" dictionary *AND* by index into the
                # variadic "*args" tuple.
                else:
                    # String evaluating to this parameter's current value when
                    # passed positionally.
                    func_arg_value_pos_expr = 'args[{!r}]'.format(
                        func_arg_index)

                    func_body += '''
    if not (
        isinstance({arg_value_pos_expr}, {arg_type_expr})
        if {arg_index} < len(args) else
        isinstance({arg_value_key_expr}, {arg_type_expr})
        if {arg_name!r} in kwargs else True):
            raise TypeError(
                '{func_name} parameter {arg_name}={{}} not of {{!r}}'.format(
                {arg_value_pos_expr} if {arg_index} < len(args) else {arg_value_key_expr},
                {arg_type_expr}))
'''.format(
                        func_name=func_name,
                        arg_name=func_arg.name,
                        arg_index=func_arg_index,
                        arg_type_expr=func_arg_type_expr,
                        arg_value_key_expr=func_arg_value_key_expr,
                        arg_value_pos_expr=func_arg_value_pos_expr,
                    )

        # If this callable's return value is both annotated and non-ignorable
        # for purposes of type checking, type check this value.
        if func_sig.return_annotation not in _RETURN_ANNOTATION_IGNORED:
            # Validate this annotation.
            _check_type_annotation(
                annotation=func_sig.return_annotation,
                label='{} return type'.format(func_name))

            # Strings evaluating to this parameter's annotated type and
            # currently passed value, as above.
            func_return_type_expr = (
                "__beartype_func.__annotations__['return']")

            # Call this callable, type check the returned value, and return this
            # value from this wrapper.
            func_body += '''
    return_value = __beartype_func(*args, **kwargs)
    if not isinstance(return_value, {return_type}):
        raise TypeError(
            '{func_name} return value {{}} not of {{!r}}'.format(
                return_value, {return_type}))
    return return_value
'''.format(func_name=func_name, return_type=func_return_type_expr)
        # Else, call this callable and return this value from this wrapper.
        else:
            func_body += '''
    return __beartype_func(*args, **kwargs)
'''

        # Dictionary mapping from local attribute name to value. For efficiency,
        # only those local attributes explicitly required in the body of this
        # wrapper are copied from the current namespace. (See below.)
        local_attrs = {'__beartype_func': func}

        # Dynamically define this wrapper as a closure of this decorator. For
        # obscure and presumably uninteresting reasons, Python fails to locally
        # declare this closure when the locals() dictionary is passed; to
        # capture this closure, a local dictionary must be passed instead.
        exec(func_body, globals(), local_attrs)

        # Return this wrapper.
        return local_attrs['func_beartyped']

    _PARAMETER_KIND_IGNORED = {
        Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD,
    }
    '''
    Set of all `inspect.Parameter.kind` constants to be ignored during
    annotation- based type checking in the `@beartype` decorator.

    This includes:

    * Constants specific to variadic parameters (e.g., `*args`, `**kwargs`).
      Variadic parameters cannot be annotated and hence cannot be type checked.
    * Constants specific to positional-only parameters, which apply to non-pure-
      Python callables (e.g., defined by C extensions). The `@beartype`
      decorator applies _only_ to pure-Python callables, which provide no
      syntactic means of specifying positional-only parameters.
    '''

    _RETURN_ANNOTATION_IGNORED = {Signature.empty, None}
    '''
    Set of all annotations for return values to be ignored during annotation-
    based type checking in the `@beartype` decorator.

    This includes:

    * `Signature.empty`, signifying a callable whose return value is _not_
      annotated.
    * `None`, signifying a callable returning no value. By convention, callables
      returning no value are typically annotated to return `None`. Technically,
      callables whose return values are annotated as `None` _could_ be
      explicitly checked to return `None` rather than a none-`None` value. Since
      return values are safely ignorable by callers, however, there appears to
      be little real-world utility in enforcing this constraint.
    '''

    def _check_type_annotation(annotation: object, label: str) -> None:
        '''
        Validate the passed annotation to be a valid type supported by the
        `@beartype` decorator.

        Parameters
        ----------
        annotation : object
            Annotation to be validated.
        label : str
            Human-readable label describing this annotation, interpolated into
            exceptions raised by this function.

        Raises
        ----------
        TypeError
            If this annotation is neither a new-style class nor a tuple of
            new-style classes.
        '''

        # If this annotation is a tuple, raise an exception if any member of
        # this tuple is not a new-style class. Note that the "__name__"
        # attribute tested below is not defined by old-style classes and hence
        # serves as a helpful means of identifying new-style classes.
        if isinstance(annotation, tuple):
            for member in annotation:
                if not (
                        isinstance(member, type) and hasattr(member, '__name__')):
                    raise TypeError(
                        '{} tuple member {} not a new-style class'.format(
                            label, member))
        # Else if this annotation is not a new-style class, raise an exception.
        elif not (
                isinstance(annotation, type) and hasattr(annotation, '__name__')):
            raise TypeError(
                '{} {} neither a new-style class nor '
                'tuple of such classes'.format(label, annotation))

# Else, the active Python interpreter is optimized. In this case, disable type
# checking by reducing this decorator to the identity decorator.
else:
    def beartype(func: callable) -> callable:
        return func


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h // nrows, -1, nrows, ncols)
               .swapaxes(1, 2)
               .reshape(h, w))

#    def export_to_plot(self):
#        """Export the domain and the correspondent values for the reconstruction.
#
#        Return
#        ------
#        (x, y) : tuple of ndarrays
#            Contain the x and y coordinates of the domain
#        reconstructed : ndarray
#            the reconstrued values of the form for the domain pts
#        """
#        n_x, n_y = self.function_space.mesh.n_x, self.function_space.mesh.n_y
# print(self.basis.eta)
#        x, y = self.function_space.mesh.mapping(self.basis.xi, self.basis.eta)
##        print("shape x :", np.shape(x))
##        print("shape self rc :", np.shape(self.reconstructed))
##        print("shape y :", np.shape(y))
#        for i in range(n_x * n_y):
#            cs = plt.contourf(x[:, :, i], y[:, :, i], self.reconstructed[i,
#                                                                         :].reshape(np.shape(x[:, :, i]), order='F'))
#        plt.colorbar(cs)
#        plt.show()
#        print("The max:", np.max(self.reconstructed))
#        print("The min:", np.min(self.reconstructed))
##        m_y, m_x = np.shape(x)[:-1]
# reshape 3d array into a 2d ones
## m_y, m_x = np.shape(x[:,:,1])
##
##        x = x.swapaxes(0,2).reshape(m_y * n_y, m_x * n_x, order='F')
##        y = y.swapaxes(0,2).reshape(m_y * n_y, m_x * n_x, order='F')
## reconstructed = self.reconstructed.reshape(m_y, m_x, n_y * n_x, order='F')
##        reconstructed = self.reconstructed.reshape(m_y * n_y, m_x * n_x, order='F')
## reconstructed = np.zeros( shape = (np.shape(x)))
# for i in range(n_x * n_y):
## reconstructed[:,:,i] = np.reshape(self.reconstructed[i,:],(m_y, m_x), order = 'F')
###
## x = x.reshape( m_y*m_x, n_x*n_y , order='F')
## y = y.reshape( m_y*m_x, n_x*n_y , order='F')
## reconstructed = reconstructed.reshape(m_y*m_x, n_x*n_y , order='F')
# return (x, y), reconstructed


@beartype
def sum(a, b: int) -> int:
    return a + b


def function_helper(x, y):
    return x**2 + y**2


def runInParallel(fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


if __name__ == '__main__':
    print(sum(3.1, 2))
