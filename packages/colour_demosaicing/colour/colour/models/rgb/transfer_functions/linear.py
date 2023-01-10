# -*- coding: utf-8 -*-
"""
Linear Colour Component Transfer Function
=========================================

Defines the linear encoding / decoding colour component transfer function
related objects:

- :func:`colour.linear_function`
"""

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'linear_function',
]


def linear_function(a):
    """
    Defines a typical linear encoding / decoding function, essentially a
    pass-through function.

    Parameters
    ----------
    a : numeric or array_like
        Array to encode / decode.

    Returns
    -------
    numeric or ndarray
        Encoded / decoded array.

    Examples
    --------
    >>> linear_function(0.18)
    0.18
    """

    return a
