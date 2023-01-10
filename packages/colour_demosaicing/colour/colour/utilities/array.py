# -*- coding: utf-8 -*-
"""
Array Utilities
===============

Defines array utilities objects.

References
----------
-   :cite:`Castro2014a` : Castro, S. (2014). Numpy: Fastest way of computing
    diagonal for each row of a 2d array. Retrieved August 22, 2014, from
    http://stackoverflow.com/questions/26511401/\
numpy-fastest-way-of-computing-diagonal-for-each-row-of-a-2d-array/\
26517247#26517247
-   :cite:`Yorke2014a` : Yorke, R. (2014). Python: Change format of np.array or
    allow tolerance in in1d function. Retrieved March 27, 2015, from
    http://stackoverflow.com/a/23521245/931625
"""

import numpy as np
import sys
from collections.abc import ValuesView
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from operator import add, mul, pow, sub, truediv

from colour.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE, EPSILON
from colour.utilities import attest, suppress_warnings, validate_method

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MixinDataclassArray',
    'as_array',
    'as_int_array',
    'as_float_array',
    'as_numeric',
    'as_int',
    'as_float',
    'set_float_precision',
    'set_int_precision',
    'closest_indexes',
    'closest',
    'interval',
    'is_uniform',
    'in_array',
    'tstack',
    'tsplit',
    'row_as_diagonal',
    'orient',
    'centroid',
    'fill_nan',
    'has_only_nan',
    'ndarray_write',
    'zeros',
    'ones',
    'full',
    'index_along_last_axis',
]


class MixinDataclassArray:
    """
    A mixin providing conversion methods for :class:`dataclass` conversion to
    :class:`ndarray` class and mathematical operations.

    Methods
    -------
    -   :meth:`~colour.utilities.MixinDataclassArray.__array__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__iadd__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__add__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__isub__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__sub__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__imul__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__mul__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__idiv__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__div__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__ipow__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__pow__`
    -   :meth:`~colour.utilities.MixinDataclassArray.arithmetical_operation`

    """

    def __array__(self, *args, **kwargs):
        """
        Implements support for *dataclass_like* conversion to :class:`ndarray`
        class.

        A field set to *None* will be filled with `np.nan` according to the
        shape of the first field not set with *None*.

        Other Parameters
        ----------------
        \\*args : list, optional
            Arguments.
        \\**kwargs : dict, optional
            Keywords arguments.

        Returns
        -------
        ndarray
            *dataclass_like* converted to `ndarray`.
        """

        field_values = {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

        default = None
        for field, value in field_values.items():
            if value is not None:
                default = full(as_float_array(value).shape, np.nan)
                break

        return tstack([
            value if value is not None else default
            for value in field_values.values()
        ], *args, **kwargs)

    def __add__(self, a):
        """
        Implements support for addition.

        Parameters
        ----------
        a : numeric or array_like or dataclass_like
            :math:`a` variable to add.

        Returns
        -------
        dataclass_like
            Variable added *dataclass_like*.
        """

        return self.arithmetical_operation(a, '+')

    def __iadd__(self, a):
        """
        Implements support for in-place addition.

        Parameters
        ----------
        a : numeric or array_like or dataclass_like
            :math:`a` variable to add in-place.

        Returns
        -------
        dataclass_like
            In-place variable added *dataclass_like*.
        """

        return self.arithmetical_operation(a, '+', True)

    def __sub__(self, a):
        """
        Implements support for subtraction.

        Parameters
        ----------
        a : numeric or array_like or dataclass_like
            :math:`a` variable to subtract.

        Returns
        -------
        dataclass_like
            Variable subtracted *dataclass_like*.
        """

        return self.arithmetical_operation(a, '-')

    def __isub__(self, a):
        """
        Implements support for in-place subtraction.

        Parameters
        ----------
        a : numeric or array_like or dataclass_like
            :math:`a` variable to subtract in-place.

        Returns
        -------
        dataclass_like
            In-place variable subtracted *dataclass_like*.
        """

        return self.arithmetical_operation(a, '-', True)

    def __mul__(self, a):
        """
        Implements support for multiplication.

        Parameters
        ----------
        a : numeric or array_like or dataclass_like
            :math:`a` variable to multiply by.

        Returns
        -------
        dataclass_like
            Variable multiplied *dataclass_like*.
        """

        return self.arithmetical_operation(a, '*')

    def __imul__(self, a):
        """
        Implements support for in-place multiplication.

        Parameters
        ----------
        a : numeric or array_like or dataclass_like
            :math:`a` variable to multiply by in-place.

        Returns
        -------
        dataclass_like
            In-place variable multiplied *dataclass_like*.
        """

        return self.arithmetical_operation(a, '*', True)

    def __div__(self, a):
        """
        Implements support for division.

        Parameters
        ----------
        a : numeric or array_like or dataclass_like
            :math:`a` variable to divide by.

        Returns
        -------
        dataclass_like
            Variable divided *dataclass_like*.
        """

        return self.arithmetical_operation(a, '/')

    def __idiv__(self, a):
        """
        Implements support for in-place division.

        Parameters
        ----------
        a : numeric or array_like or dataclass_like
            :math:`a` variable to divide by in-place.

        Returns
        -------
        dataclass_like
            In-place variable divided *dataclass_like*.
        """

        return self.arithmetical_operation(a, '/', True)

    __itruediv__ = __idiv__
    __truediv__ = __div__

    def __pow__(self, a):
        """
        Implements support for exponentiation.

        Parameters
        ----------
        a : numeric or array_like or dataclass_like
            :math:`a` variable to exponentiate by.

        Returns
        -------
        dataclass_like
            Variable exponentiated *dataclass_like*.
        """

        return self.arithmetical_operation(a, '**')

    def __ipow__(self, a):
        """
        Implements support for in-place exponentiation.

        Parameters
        ----------
        a : numeric or array_like or dataclass_like
            :math:`a` variable to exponentiate by in-place.

        Returns
        -------
        dataclass_like
            In-place variable exponentiated *dataclass_like*.
        """

        return self.arithmetical_operation(a, '**', True)

    def arithmetical_operation(self, a, operation, in_place=False):
        """
        Performs given arithmetical operation with :math:`a` operand on the
        *dataclass_like*.

        Parameters
        ----------
        a : numeric or ndarray or dataclass_like
            Operand.
        operation : object
            Operation to perform.
        in_place : bool, optional
            Operation happens in place.

        Returns
        -------
        dataclass_like
            *Dataclass_like* with arithmetical operation performed.
        """

        operation = {
            '+': add,
            '-': sub,
            '*': mul,
            '/': truediv,
            '**': pow,
        }[operation]

        if is_dataclass(a):
            a = as_float_array(a)

        field_values = tsplit(operation(as_float_array(self), a))
        field_values = {
            field.name: field_values[i]
            for i, field in enumerate(fields(self))
        }
        field_values.update(
            {field.name: None
             for field in fields(self) if field is None})

        dataclass = replace(self, **field_values)

        if in_place:
            for field in fields(self):
                setattr(self, field.name, getattr(dataclass, field.name))
            return self
        else:
            return dataclass


def as_array(a, dtype=None):
    """
    Converts given :math:`a` variable to *ndarray* using given type.

    Parameters
    ----------
    a : object
        Variable to convert.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *ndarray*.

    Examples
    --------
    >>> as_array([1, 2, 3])
    array([ 1.,  2.,  3.])
    >>> as_array([1, 2, 3], dtype=DEFAULT_INT_DTYPE)  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    # TODO: Remove when https://github.com/numpy/numpy/issues/5718 is
    # addressed.
    if isinstance(a, ValuesView):
        a = list(a)

    return np.asarray(a, dtype)


def as_int_array(a, dtype=None):
    """
    Converts given :math:`a` variable to *ndarray* using given type.

    Parameters
    ----------
    a : object
        Variable to convert.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *ndarray*.

    Examples
    --------
    >>> as_int_array([1.0, 2.0, 3.0])  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    """

    if dtype is None:
        dtype = DEFAULT_INT_DTYPE

    attest(
        dtype in np.sctypes['int'],
        '"dtype" must be one of the following types: {0}'.format(
            np.sctypes['int']))

    return as_array(a, dtype)


def as_float_array(a, dtype=None):
    """
    Converts given :math:`a` variable to *ndarray* using given type.

    Parameters
    ----------
    a : object
        Variable to convert.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *ndarray*.

    Examples
    --------
    >>> as_float_array([1, 2, 3])
    array([ 1.,  2.,  3.])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    attest(
        dtype in np.sctypes['float'],
        '"dtype" must be one of the following types: {0}'.format(
            np.sctypes['float']))

    return as_array(a, dtype)


def as_numeric(a, dtype=None):
    """
    Converts given :math:`a` variable to *numeric*. In the event where
    :math:`a` cannot be converted, it is passed as is.

    Parameters
    ----------
    a : object
        Variable to convert.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *numeric*.

    Examples
    --------
    >>> as_numeric(np.array([1]))
    1.0
    >>> as_numeric(np.arange(10))
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    try:
        return dtype(a)
    except (TypeError, ValueError):
        return a


def as_int(a, dtype=None):
    """
    Attempts to converts given :math:`a` variable to *int* using given type.

    Parameters
    ----------
    a : object
        Variable to convert.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute. In the event where
        :math:`a` cannot be converted, it is converted to *ndarray* using the
        type defined by :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *numeric*.

    Warnings
    --------
    The behaviour of this definition is different than
    :func:`colour.utilities.as_numeric` definition when it comes to conversion
    failure: the former will forcibly convert :math:`a` variable to *ndarray*
    using the type defined by :attr:`colour.constant.DEFAULT_INT_DTYPE`
    attribute while the later will pass the :math:`a` variable as is.

    Examples
    --------
    >>> as_int(np.array([1]))
    1
    >>> as_int(np.arange(10))  # doctest: +ELLIPSIS
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]...)
    """

    if dtype is None:
        dtype = DEFAULT_INT_DTYPE

    attest(
        dtype in np.sctypes['int'],
        '"dtype" must be one of the following types: {0}'.format(
            np.sctypes['int']))
    try:
        # TODO: Change to "DEFAULT_INT_DTYPE" when and if
        # https://github.com/numpy/numpy/issues/11956 is addressed.
        return int(a)
    except TypeError:
        return as_int_array(a, dtype)


def as_float(a, dtype=None):
    """
    Converts given :math:`a` variable to *numeric* using given type.

    Parameters
    ----------
    a : object
        Variable to convert.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute. In the event where
        :math:`a` cannot be converted, it is converted to *ndarray* using the
        type defined by :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *numeric*.

    Warnings
    --------
    The behaviour of this definition is different than
    :func:`colour.utilities.as_numeric` definition when it comes to conversion
    failure: the former will forcibly convert :math:`a` variable to *ndarray*
    using the type defined by :attr:`colour.constant.DEFAULT_FLOAT_DTYPE`
    attribute while the later will pass the :math:`a` variable as is.

    Examples
    --------
    >>> as_float(np.array([1]))
    1.0
    >>> as_float(np.arange(10))
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    attest(
        dtype in np.sctypes['float'],
        '"dtype" must be one of the following types: {0}'.format(
            np.sctypes['float']))

    return dtype(a)


def set_float_precision(dtype=DEFAULT_FLOAT_DTYPE):
    """
    Sets *Colour* float precision by setting
    :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute with given type
    wherever the attribute is imported.

    Parameters
    ----------
    dtype : object
        Type to set :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` with.

    Warnings
    --------
    Changing float precision might result in various *Colour* functionality
    breaking entirely: https://github.com/numpy/numpy/issues/6860. With great
    power comes great responsibility.

    Notes
    -----
    -   It is possible to define the float precision at import time by setting
        the *COLOUR_SCIENCE__FLOAT_PRECISION* environment variable, for example
        `set COLOUR_SCIENCE__FLOAT_PRECISION=float32`.
    -   Some definition returning a single-scalar ndarray might not honour the
        given float precision: https://github.com/numpy/numpy/issues/16353

    Examples
    --------
    >>> as_float_array(np.ones(3)).dtype
    dtype('float64')
    >>> set_float_precision(np.float16)
    >>> as_float_array(np.ones(3)).dtype
    dtype('float16')
    >>> set_float_precision(np.float64)
    >>> as_float_array(np.ones(3)).dtype
    dtype('float64')
    """

    with suppress_warnings(colour_usage_warnings=True):
        for module in sys.modules.values():
            if not hasattr(module, 'DEFAULT_FLOAT_DTYPE'):
                continue

            setattr(module, 'DEFAULT_FLOAT_DTYPE', dtype)


def set_int_precision(dtype=DEFAULT_INT_DTYPE):
    """
    Sets *Colour* integer precision by setting
    :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute with given type
    wherever the attribute is imported.

    Parameters
    ----------
    dtype : object
        Type to set :attr:`colour.constant.DEFAULT_INT_DTYPE` with.

    Notes
    -----
    -   It is possible to define the int precision at import time by setting
        the *COLOUR_SCIENCE__INT_PRECISION* environment variable, for example
        `set COLOUR_SCIENCE__INT_PRECISION=int32`.

    Warnings
    --------
    This definition is mostly given for consistency purposes with
    :func:`colour.utilities.set_float_precision` definition but contrary to the
    latter, changing integer precision will almost certainly completely break
    *Colour*. With great power comes great responsibility.

    Examples
    --------
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int64')
    >>> set_int_precision(np.int32)
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int32')
    >>> set_int_precision(np.int64)
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int64')
    """

    # TODO: Investigate behaviour on Windows.
    with suppress_warnings(colour_usage_warnings=True):
        for name, module in sys.modules.items():
            if not hasattr(module, 'DEFAULT_INT_DTYPE'):
                continue

            setattr(module, 'DEFAULT_INT_DTYPE', dtype)


def closest_indexes(a, b):
    """
    Returns the :math:`a` variable closest element indexes to reference
    :math:`b` variable elements.

    Parameters
    ----------
    a : array_like
        Variable to search for the closest element indexes.
    b : numeric
        Reference variable.

    Returns
    -------
    numeric
        Closest :math:`a` variable element indexes.

    Examples
    --------
    >>> a = np.array([24.31357115, 63.62396289, 55.71528816,
    ...               62.70988028, 46.84480573, 25.40026416])
    >>> print(closest_indexes(a, 63))
    [3]
    >>> print(closest_indexes(a, [63, 25]))
    [3 5]
    """

    a = np.ravel(a)[:, np.newaxis]
    b = np.ravel(b)[np.newaxis, :]

    return np.abs(a - b).argmin(axis=0)


def closest(a, b):
    """
    Returns the :math:`a` variable closest elements to reference :math:`b`
    variable elements.

    Parameters
    ----------
    a : array_like
        Variable to search for the closest elements.
    b : numeric
        Reference variable.

    Returns
    -------
    numeric
        Closest :math:`a` variable elements.

    Examples
    --------
    >>> a = np.array([24.31357115, 63.62396289, 55.71528816,
    ...               62.70988028, 46.84480573, 25.40026416])
    >>> closest(a, 63)
    array([ 62.70988028])
    >>> closest(a, [63, 25])
    array([ 62.70988028,  25.40026416])
    """

    a = np.array(a)

    return a[closest_indexes(a, b)]


def interval(distribution, unique=True):
    """
    Returns the interval size of given distribution.

    Parameters
    ----------
    distribution : array_like
        Distribution to retrieve the interval.
    unique : bool, optional
        Whether to return unique intervals if  the distribution is
        non-uniformly spaced or the complete intervals

    Returns
    -------
    ndarray
        Distribution interval.

    Examples
    --------
    Uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 5])
    >>> interval(y)
    array([ 1.])
    >>> interval(y, False)
    array([ 1.,  1.,  1.,  1.])

    Non-uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 8])
    >>> interval(y)
    array([ 1.,  4.])
    >>> interval(y, False)
    array([ 1.,  1.,  1.,  4.])
    """

    distribution = as_float_array(distribution)
    i = np.arange(distribution.size - 1)

    differences = np.abs(distribution[i + 1] - distribution[i])
    if unique:
        return np.unique(differences)
    else:
        return differences


def is_uniform(distribution):
    """
    Returns if given distribution is uniform.

    Parameters
    ----------
    distribution : array_like
        Distribution to check for uniformity.

    Returns
    -------
    bool
        Is distribution uniform.

    Examples
    --------
    Uniformly spaced variable:

    >>> a = np.array([1, 2, 3, 4, 5])
    >>> is_uniform(a)
    True

    Non-uniformly spaced variable:

    >>> a = np.array([1, 2, 3.1415, 4, 5])
    >>> is_uniform(a)
    False
    """

    return True if interval(distribution).size == 1 else False


def in_array(a, b, tolerance=EPSILON):
    """
    Tests whether each element of an array is also present in a second array
    within given tolerance.

    Parameters
    ----------
    a : array_like
        Array to test the elements from.
    b : array_like
        The values against which to test each value of array *a*.
    tolerance : numeric, optional
        Tolerance value.

    Returns
    -------
    ndarray
        A boolean array with *a* shape describing whether an element of *a* is
        present in *b* within given tolerance.

    References
    ----------
    :cite:`Yorke2014a`

    Examples
    --------
    >>> a = np.array([0.50, 0.60])
    >>> b = np.linspace(0, 10, 101)
    >>> np.in1d(a, b)
    array([ True, False], dtype=bool)
    >>> in_array(a, b)
    array([ True,  True], dtype=bool)
    """

    a = as_float_array(a)
    b = as_float_array(b)

    d = np.abs(np.ravel(a) - b[..., np.newaxis])

    return np.any(d <= tolerance, axis=0).reshape(a.shape)


def tstack(a, dtype=None):
    """
    Stacks arrays in sequence along the last axis (tail).

    Rebuilds arrays divided by :func:`colour.utilities.tsplit`.

    Parameters
    ----------
    a : array_like
        Array to perform the stacking.
    dtype : object
        Type to use for initial conversion to *ndarray*, default to the type
        defined by :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> a = 0
    >>> tstack([a, a, a])
    array([ 0.,  0.,  0.])
    >>> a = np.arange(0, 6)
    >>> tstack([a, a, a])
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.],
           [ 2.,  2.,  2.],
           [ 3.,  3.,  3.],
           [ 4.,  4.,  4.],
           [ 5.,  5.,  5.]])
    >>> a = np.reshape(a, (1, 6))
    >>> tstack([a, a, a])
    array([[[ 0.,  0.,  0.],
            [ 1.,  1.,  1.],
            [ 2.,  2.,  2.],
            [ 3.,  3.,  3.],
            [ 4.,  4.,  4.],
            [ 5.,  5.,  5.]]])
    >>> a = np.reshape(a, (1, 1, 6))
    >>> tstack([a, a, a])
    array([[[[ 0.,  0.,  0.],
             [ 1.,  1.,  1.],
             [ 2.,  2.,  2.],
             [ 3.,  3.,  3.],
             [ 4.,  4.,  4.],
             [ 5.,  5.,  5.]]]])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    a = as_array(a, dtype)

    return np.concatenate([x[..., np.newaxis] for x in a], axis=-1)


def tsplit(a, dtype=None):
    """
    Splits arrays in sequence along the last axis (tail).

    Parameters
    ----------
    a : array_like
        Array to perform the splitting.
    dtype : object
        Type to use for initial conversion to *ndarray*, default to the type
        defined by :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> a = np.array([0, 0, 0])
    >>> tsplit(a)
    array([ 0.,  0.,  0.])
    >>> a = np.array(
    ...     [[0, 0, 0],
    ...      [1, 1, 1],
    ...      [2, 2, 2],
    ...      [3, 3, 3],
    ...      [4, 4, 4],
    ...      [5, 5, 5]]
    ... )
    >>> tsplit(a)
    array([[ 0.,  1.,  2.,  3.,  4.,  5.],
           [ 0.,  1.,  2.,  3.,  4.,  5.],
           [ 0.,  1.,  2.,  3.,  4.,  5.]])
    >>> a = np.array(
    ...     [[[0, 0, 0],
    ...       [1, 1, 1],
    ...       [2, 2, 2],
    ...       [3, 3, 3],
    ...       [4, 4, 4],
    ...       [5, 5, 5]]]
    ... )
    >>> tsplit(a)
    array([[[ 0.,  1.,  2.,  3.,  4.,  5.]],
    <BLANKLINE>
           [[ 0.,  1.,  2.,  3.,  4.,  5.]],
    <BLANKLINE>
           [[ 0.,  1.,  2.,  3.,  4.,  5.]]])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    a = as_array(a, dtype)

    return np.array([a[..., x] for x in range(a.shape[-1])])


def row_as_diagonal(a):
    """
    Returns the per row diagonal matrices of the given array.

    Parameters
    ----------
    a : array_like
        Array to perform the diagonal matrices computation.

    Returns
    -------
    ndarray

    References
    ----------
    :cite:`Castro2014a`

    Examples
    --------
    >>> a = np.array(
    ...     [[0.25891593, 0.07299478, 0.36586996],
    ...       [0.30851087, 0.37131459, 0.16274825],
    ...       [0.71061831, 0.67718718, 0.09562581],
    ...       [0.71588836, 0.76772047, 0.15476079],
    ...       [0.92985142, 0.22263399, 0.88027331]]
    ... )
    >>> row_as_diagonal(a)
    array([[[ 0.25891593,  0.        ,  0.        ],
            [ 0.        ,  0.07299478,  0.        ],
            [ 0.        ,  0.        ,  0.36586996]],
    <BLANKLINE>
           [[ 0.30851087,  0.        ,  0.        ],
            [ 0.        ,  0.37131459,  0.        ],
            [ 0.        ,  0.        ,  0.16274825]],
    <BLANKLINE>
           [[ 0.71061831,  0.        ,  0.        ],
            [ 0.        ,  0.67718718,  0.        ],
            [ 0.        ,  0.        ,  0.09562581]],
    <BLANKLINE>
           [[ 0.71588836,  0.        ,  0.        ],
            [ 0.        ,  0.76772047,  0.        ],
            [ 0.        ,  0.        ,  0.15476079]],
    <BLANKLINE>
           [[ 0.92985142,  0.        ,  0.        ],
            [ 0.        ,  0.22263399,  0.        ],
            [ 0.        ,  0.        ,  0.88027331]]])
    """

    a = np.expand_dims(a, -2)

    return np.eye(a.shape[-1]) * a


def orient(a, orientation):
    """
    Orient given array according to given ``orientation`` value.

    Parameters
    ----------
    a : array_like
        Array to perform the orientation onto.
    orientation : str, optional
        **{'Flip', 'Flop', '90 CW', '90 CCW', '180'}**
        Orientation to perform.

    Returns
    -------
    ndarray
        Oriented array.

    Examples
    --------
    >>> a = np.tile(np.arange(5), (5, 1))
    >>> a
    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])
    >>> orient(a, '90 CW')
    array([[0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]])
    >>> orient(a, 'Flip')
    array([[4, 3, 2, 1, 0],
           [4, 3, 2, 1, 0],
           [4, 3, 2, 1, 0],
           [4, 3, 2, 1, 0],
           [4, 3, 2, 1, 0]])
    """

    orientation = validate_method(orientation,
                                  ['Flip', 'Flop', '90 CW', '90 CCW', '180'])

    if orientation == 'flip':
        return np.fliplr(a)
    elif orientation == 'flop':
        return np.flipud(a)
    elif orientation == '90 cw':
        return np.rot90(a, 3)
    elif orientation == '90 ccw':
        return np.rot90(a)
    elif orientation == '180':
        return np.rot90(a, 2)


def centroid(a):
    """
    Computes the centroid indexes of given :math:`a` array.

    Parameters
    ----------
    a : array_like
        :math:`a` array to compute the centroid indexes.

    Returns
    -------
    ndarray
        :math:`a` array centroid indexes.

    Examples
    --------
    >>> a = np.tile(np.arange(0, 5), (5, 1))
    >>> centroid(a)  # doctest: +ELLIPSIS
    array([2, 3]...)
    """

    a = as_float_array(a)

    a_s = np.sum(a)

    ranges = [np.arange(0, a.shape[i]) for i in range(a.ndim)]
    coordinates = np.meshgrid(*ranges)

    a_ci = []
    for axis in coordinates:
        axis = np.transpose(axis)
        # Aligning axis for N-D arrays where N is normalised to
        # range [3, :math:`\\\infty`]
        for i in range(axis.ndim - 2, 0, -1):
            axis = np.rollaxis(axis, i - 1, axis.ndim)

        a_ci.append(np.sum(axis * a) // a_s)

    return np.array(a_ci).astype(DEFAULT_INT_DTYPE)


def fill_nan(a, method='Interpolation', default=0):
    """
    Fills given array NaNs according to given method.

    Parameters
    ----------
    a : array_like
        Array to fill the NaNs of.
    method : str
        **{'Interpolation', 'Constant'}**,
        *Interpolation* method linearly interpolates through the NaNs,
        *Constant* method replaces NaNs with ``default``.
    default : numeric
        Value to use with the *Constant* method.

    Returns
    -------
    ndarray
        NaNs filled array.

    Examples
    --------
    >>> a = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    >>> fill_nan(a)
    array([ 0.1,  0.2,  0.3,  0.4,  0.5])
    >>> fill_nan(a, method='Constant')
    array([ 0.1,  0.2,  0. ,  0.4,  0.5])
    """

    a = np.copy(a)
    method = validate_method(method, ['Interpolation', 'Constant'])

    mask = np.isnan(a)

    if method == 'interpolation':
        a[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), a[~mask])
    elif method == 'constant':
        a[mask] = default

    return a


def has_only_nan(a):
    """
    Returns whether given array :math:`a` contains only nan values.

    Parameters
    ----------
    a : array_like
        :math:`a` array to check whether it contains only nan values.

    Returns
    -------
    bool
        Whether array :math:`a` contains only nan values.

    Examples
    --------
    >>> has_only_nan(None)
    True
    >>> has_only_nan([None, None])
    True
    >>> has_only_nan([True, None])
    False
    >>> has_only_nan([0.1, np.nan, 0.3])
    False
    """

    return np.all(np.isnan(as_float_array(a)))


@contextmanager
def ndarray_write(a):
    """
    A context manager setting given array writeable to perform an operation
    and then read-only.

    Parameters
    ----------
    a : array_like
        Array to perform an operation.

    Returns
    -------
    ndarray
        Array.

    Examples
    --------
    >>> a = np.linspace(0, 1, 10)
    >>> a.setflags(write=False)
    >>> try:
    ...     a += 1
    ... except ValueError:
    ...     pass
    >>> with ndarray_write(a):
    ...     a +=1
    """

    a = as_float_array(a)

    a.setflags(write=True)

    try:
        yield a
    finally:
        a.setflags(write=False)


def zeros(shape, dtype=None, order='C'):
    """
    Simple wrapper around :func:`np.zeros` definition to create arrays with
    the active type defined by the:attr:`colour.constant.DEFAULT_FLOAT_DTYPE`
    attribute.

    Parameters
    ----------
    shape : int or array_like
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    order : str, optional
        {'C', 'F'},
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    Returns
    -------
    ndarray
        Array of given shape and type, filled with zeros.

    Examples
    --------
    >>> zeros(3)
    array([ 0.,  0.,  0.])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    return np.zeros(shape, dtype, order)


def ones(shape, dtype=None, order='C'):
    """
    Simple wrapper around :func:`np.ones` definition to create arrays with
    the active type defined by the:attr:`colour.constant.DEFAULT_FLOAT_DTYPE`
    attribute.

    Parameters
    ----------
    shape : int or array_like
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    order : str, optional
        {'C', 'F'},
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    Returns
    -------
    ndarray
        Array of given shape and type, filled with ones.

    Examples
    --------
    >>> ones(3)
    array([ 1.,  1.,  1.])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    return np.ones(shape, dtype, order)


def full(shape, fill_value, dtype=None, order='C'):
    """
    Simple wrapper around :func:`np.full` definition to create arrays with
    the active type defined by the:attr:`colour.constant.DEFAULT_FLOAT_DTYPE`
    attribute.

    Parameters
    ----------
    shape : int or array_like
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : numeric
        Fill value.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    order : str, optional
        {'C', 'F'},
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    Returns
    -------
    ndarray
        Array of given shape and type, filled with given value.

    Examples
    --------
    >>> ones(3)
    array([ 1.,  1.,  1.])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    return np.full(shape, fill_value, dtype, order)


def index_along_last_axis(a, indexes):
    """
    Reduces the dimension of an array by one, by using an array of indexes to
    to pick elements off the last axis.

    Parameters
    ----------
    a : ndarray, (Ni..., m)
        Array to be indexed.
    indexes : ndarray, (Ni...)
        Integer array with the same shape as `a` but with one dimension fewer,
        containing indices to the last dimension of `a`. All elements must be
        numbers between `0` and `m` - 1.

    Returns
    -------
    ndarray, (Ni...)
        Result of the operation.

    Raises
    ------
    ValueError
        If the arrays have incompatible shapes.
    IndexError
        If `indexes` has elements outside of the allowed range of 0 to `m` - 1
        or if it's not an integer array.

    Examples
    --------
    >>> a = np.array(
    ...     [[[0.3, 0.5, 6.9],
    ...       [3.3, 4.4, 1.6],
    ...       [4.4, 7.5, 2.3],
    ...       [2.3, 1.6, 7.4]],
    ...      [[2. , 5.9, 2.8],
    ...       [6.2, 4.9, 8.6],
    ...       [3.7, 9.7, 7.3],
    ...       [6.3, 4.3, 3.2]],
    ...      [[0.8, 1.9, 0.7],
    ...       [5.6, 4. , 1.7],
    ...       [6.7, 8.2, 1.7],
    ...       [1.2, 7.1, 1.4]],
    ...      [[4. , 4.8, 8.9],
    ...       [4. , 0.3, 6.9],
    ...       [3.5, 7.1, 4.5],
    ...       [1.4, 1.9, 1.6]]]
    ... )
    >>> indexes = np.array(
    ...     [[2, 0, 1, 1],
    ...      [2, 1, 1, 0],
    ...      [0, 0, 1, 2],
    ...      [0, 0, 1, 2]]
    ... )
    >>> index_along_last_axis(a, indexes)
    array([[ 6.9,  3.3,  7.5,  1.6],
           [ 2.8,  4.9,  9.7,  6.3],
           [ 0.8,  5.6,  8.2,  1.4],
           [ 4. ,  4. ,  7.1,  1.6]])

    This function can be used to compute the result of :func:`np.min` along
    the last axis given the corresponding :func:`np.argmin` indexes.

    >>> indexes = np.argmin(a, axis=-1)
    >>> np.array_equal(
    ...     index_along_last_axis(a, indexes),
    ...     np.min(a, axis=-1)
    ... )
    True

    In particular, this can be used to manipulate the indexes given by
    functions like :func:`np.min` before indexing the array. For example, to
    get elements directly following the smallest elements:

    >>> index_along_last_axis(a, (indexes + 1) % 3)
    array([[ 0.5,  3.3,  4.4,  7.4],
           [ 5.9,  8.6,  9.7,  6.3],
           [ 0.8,  5.6,  6.7,  7.1],
           [ 4.8,  6.9,  7.1,  1.9]])
    """

    if a.shape[:-1] != indexes.shape:
        raise ValueError('Arrays have incompatible shapes: {0} and {1}'.format(
            a.shape, indexes.shape))

    return np.take_along_axis(
        a, indexes[..., np.newaxis], axis=-1).squeeze(axis=-1)
