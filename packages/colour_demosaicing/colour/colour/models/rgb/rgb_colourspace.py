# -*- coding: utf-8 -*-
"""
RGB Colourspace and Transformations
===================================

Defines the :class:`colour.RGB_Colourspace` class for the *RGB* colourspaces
datasets from :mod:`colour.models.datasets.aces_rgb`, etc... and the following
*RGB* colourspace transformations or helper definitions:

-   :func:`colour.XYZ_to_RGB`
-   :func:`colour.RGB_to_XYZ`
-   :func:`colour.matrix_RGB_to_RGB`
-   :func:`colour.RGB_to_RGB`

References
----------
-   :cite:`InternationalElectrotechnicalCommission1999a` : International
    Electrotechnical Commission. (1999). IEC 61966-2-1:1999 - Multimedia
    systems and equipment - Colour measurement and management - Part 2-1:
    Colour management - Default RGB colour space - sRGB (p. 51).
    https://webstore.iec.ch/publication/6169
-   :cite:`Panasonic2014a` : Panasonic. (2014). VARICAM V-Log/V-Gamut (pp.
    1-7).
    http://pro-av.panasonic.net/en/varicam/common/pdf/VARICAM_V-Log_V-Gamut.pdf
"""

import numpy as np
from copy import deepcopy

from colour.adaptation import matrix_chromatic_adaptation_VonKries
from colour.algebra import matrix_dot, vector_dot
from colour.models import xy_to_XYZ, xy_to_xyY, xyY_to_XYZ
from colour.models.rgb import (
    chromatically_adapted_primaries,
    normalised_primary_matrix,
)
from colour.utilities import (
    as_float_array,
    attest,
    domain_range_scale,
    filter_kwargs,
    from_range_1,
    to_domain_1,
    is_string,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'RGB_Colourspace',
    'XYZ_to_RGB',
    'RGB_to_XYZ',
    'matrix_RGB_to_RGB',
    'RGB_to_RGB',
]


class RGB_Colourspace:
    """
    Implements support for the *RGB* colourspaces datasets from
    :mod:`colour.models.datasets.aces_rgb`, etc....

    Colour science literature related to *RGB* colourspaces and encodings
    defines their dataset using different degree of precision or rounding.
    While instances where a whitepoint is being defined with a value
    different than its canonical agreed one are rare, it is however very
    common to have normalised primary matrices rounded at different
    decimals. This can yield large discrepancies in computations.

    Such an occurrence is the *V-Gamut* colourspace white paper, that defines
    the *V-Gamut* to *ITU-R BT.709* conversion matrix as follows::

        [[ 1.806576 -0.695697 -0.110879]
         [-0.170090  1.305955 -0.135865]
         [-0.025206 -0.154468  1.179674]]

    Computing this matrix using *ITU-R BT.709* colourspace derived normalised
    primary matrix yields::

        [[ 1.8065736 -0.6956981 -0.1108786]
         [-0.1700890  1.3059548 -0.1358648]
         [-0.0252057 -0.1544678  1.1796737]]

    The latter matrix is almost equals with the former, however performing the
    same computation using *IEC 61966-2-1:1999* *sRGB* colourspace normalised
    primary matrix introduces severe disparities::

        [[ 1.8063853 -0.6956147 -0.1109453]
         [-0.1699311  1.3058387 -0.1358616]
         [-0.0251630 -0.1544899  1.1797117]]

    In order to provide support for both literature defined dataset and
    accurate computations enabling transformations without loss of precision,
    the :class:`colour.RGB_Colourspace` class provides two sets of
    transformation matrices:

        -   Instantiation transformation matrices
        -   Derived transformation matrices

    Upon instantiation, the :class:`colour.RGB_Colourspace` class stores the
    given ``matrix_RGB_to_XYZ`` and ``matrix_XYZ_to_RGB`` arguments and also
    computes their derived counterpart using the ``primaries`` and
    ``whitepoint`` arguments.

    Whether the initialisation or derived matrices are used in subsequent
    computations is dependent on the
    :attr:`colour.RGB_Colourspace.use_derived_matrix_RGB_to_XYZ` and
    :attr:`colour.RGB_Colourspace.use_derived_matrix_XYZ_to_RGB` attribute
    values.

    Parameters
    ----------
    name : str
        *RGB* colourspace name.
    primaries : array_like
        *RGB* colourspace primaries.
    whitepoint : array_like
        *RGB* colourspace whitepoint.
    whitepoint_name : str, optional
        *RGB* colourspace whitepoint name.
    matrix_RGB_to_XYZ : array_like, optional
        Transformation matrix from colourspace to *CIE XYZ* tristimulus values.
    matrix_XYZ_to_RGB : array_like, optional
        Transformation matrix from *CIE XYZ* tristimulus values to colourspace.
    cctf_encoding : object, optional
        Encoding colour component transfer function (Encoding CCTF) /
        opto-electronic transfer function (OETF / OECF) that maps estimated
        tristimulus values in a scene to :math:`R'G'B'` video component signal
        value.
    cctf_decoding : object, optional
        Decoding colour component transfer function (Decoding CCTF) /
        electro-optical transfer function (EOTF / EOCF) that maps an
        :math:`R'G'B'` video component signal value to tristimulus values at
        the display.
    use_derived_matrix_RGB_to_XYZ : bool, optional
        Whether to use the instantiation time normalised primary matrix or to
        use a computed derived normalised primary matrix.
    use_derived_matrix_XYZ_to_RGB : bool, optional
        Whether to use the instantiation time inverse normalised primary
        matrix or to use a computed derived inverse normalised primary matrix.

    Attributes
    ----------
    -   :attr:`~colour.RGB_Colourspace.name`
    -   :attr:`~colour.RGB_Colourspace.primaries`
    -   :attr:`~colour.RGB_Colourspace.whitepoint`
    -   :attr:`~colour.RGB_Colourspace.whitepoint_name`
    -   :attr:`~colour.RGB_Colourspace.matrix_RGB_to_XYZ`
    -   :attr:`~colour.RGB_Colourspace.matrix_XYZ_to_RGB`
    -   :attr:`~colour.RGB_Colourspace.cctf_encoding`
    -   :attr:`~colour.RGB_Colourspace.cctf_decoding`
    -   :attr:`~colour.RGB_Colourspace.use_derived_matrix_RGB_to_XYZ`
    -   :attr:`~colour.RGB_Colourspace.use_derived_matrix_XYZ_to_RGB`

    Methods
    -------
    -   :attr:`~colour.RGB_Colourspace.__init__`
    -   :attr:`~colour.RGB_Colourspace.__str__`
    -   :attr:`~colour.RGB_Colourspace.__repr__`
    -   :attr:`~colour.RGB_Colourspace.use_derived_transformation_matrices`
    -   :attr:`~colour.RGB_Colourspace.chromatically_adapt`
    -   :attr:`~colour.RGB_Colourspace.copy`

    Notes
    -----
    -   The normalised primary matrix defined by
        :attr:`colour.RGB_Colourspace.matrix_RGB_to_XYZ` attribute is treated
        as the prime matrix from which the inverse will be calculated as
        required by the internal derivation mechanism. This behaviour has been
        chosen in accordance with literature where commonly a *RGB* colourspace
        is defined by its normalised primary matrix as it is directly computed
        from the chosen primaries and whitepoint.

    References
    ----------
    :cite:`InternationalElectrotechnicalCommission1999a`,
    :cite:`Panasonic2014a`

    Examples
    --------
    >>> p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = np.array([0.32168, 0.33767])
    >>> matrix_RGB_to_XYZ = np.identity(3)
    >>> matrix_XYZ_to_RGB = np.identity(3)
    >>> colourspace = RGB_Colourspace('RGB Colourspace', p, whitepoint, 'ACES',
    ...                               matrix_RGB_to_XYZ, matrix_XYZ_to_RGB)
    >>> colourspace.matrix_RGB_to_XYZ
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> colourspace.matrix_XYZ_to_RGB
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> colourspace.use_derived_transformation_matrices(True)
    True
    >>> colourspace.matrix_RGB_to_XYZ  # doctest: +ELLIPSIS
    array([[  9.5255239...e-01,   0.0000000...e+00,   9.3678631...e-05],
           [  3.4396645...e-01,   7.2816609...e-01,  -7.2132546...e-02],
           [  0.0000000...e+00,   0.0000000...e+00,   1.0088251...e+00]])
    >>> colourspace.matrix_XYZ_to_RGB  # doctest: +ELLIPSIS
    array([[  1.0498110...e+00,   0.0000000...e+00,  -9.7484540...e-05],
           [ -4.9590302...e-01,   1.3733130...e+00,   9.8240036...e-02],
           [  0.0000000...e+00,   0.0000000...e+00,   9.9125201...e-01]])
    >>> colourspace.use_derived_matrix_RGB_to_XYZ = False
    >>> colourspace.matrix_RGB_to_XYZ
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> colourspace.use_derived_matrix_XYZ_to_RGB = False
    >>> colourspace.matrix_XYZ_to_RGB
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    """

    def __init__(self,
                 name,
                 primaries,
                 whitepoint,
                 whitepoint_name=None,
                 matrix_RGB_to_XYZ=None,
                 matrix_XYZ_to_RGB=None,
                 cctf_encoding=None,
                 cctf_decoding=None,
                 use_derived_matrix_RGB_to_XYZ=False,
                 use_derived_matrix_XYZ_to_RGB=False):
        self._derived_matrix_RGB_to_XYZ = None
        self._derived_matrix_XYZ_to_RGB = None

        self._name = None
        self.name = name
        self._primaries = None
        self.primaries = primaries
        self._whitepoint = None
        self.whitepoint = whitepoint
        self._whitepoint_name = None
        self.whitepoint_name = whitepoint_name
        self._matrix_RGB_to_XYZ = None
        self.matrix_RGB_to_XYZ = matrix_RGB_to_XYZ
        self._matrix_XYZ_to_RGB = None
        self.matrix_XYZ_to_RGB = matrix_XYZ_to_RGB
        self._cctf_encoding = None
        self.cctf_encoding = cctf_encoding
        self._cctf_decoding = None
        self.cctf_decoding = cctf_decoding
        self._use_derived_matrix_RGB_to_XYZ = False
        self.use_derived_matrix_RGB_to_XYZ = use_derived_matrix_RGB_to_XYZ
        self._use_derived_matrix_XYZ_to_RGB = False
        self.use_derived_matrix_XYZ_to_RGB = use_derived_matrix_XYZ_to_RGB

    @property
    def name(self):
        """
        Getter and setter property for the name.

        Parameters
        ----------
        value : str
            Value to set the name with.

        Returns
        -------
        str
            *RGB* colourspace name.
        """

        return self._name

    @name.setter
    def name(self, value):
        """
        Setter for the **self.name** property.
        """

        if value is not None:
            attest(
                is_string(value),
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'name', value))

        self._name = value

    @property
    def primaries(self):
        """
        Getter and setter property for the primaries.

        Parameters
        ----------
        value : array_like
            Value to set the primaries with.

        Returns
        -------
        array_like
            *RGB* colourspace primaries.
        """

        return self._primaries

    @primaries.setter
    def primaries(self, value):
        """
        Setter for the **self.primaries** property.
        """

        if value is not None:
            value = np.reshape(value, (3, 2))
        self._primaries = value

        self._derive_transformation_matrices()

    @property
    def whitepoint(self):
        """
        Getter and setter property for the whitepoint.

        Parameters
        ----------
        value : array_like
            Value to set the whitepoint with.

        Returns
        -------
        array_like
            *RGB* colourspace whitepoint.
        """

        return self._whitepoint

    @whitepoint.setter
    def whitepoint(self, value):
        """
        Setter for the **self.whitepoint** property.
        """

        if value is not None:
            attest(
                isinstance(value, (tuple, list, np.ndarray, np.matrix)),
                '"{0}" attribute: "{1}" is not a "tuple", "list", "ndarray" '
                'or "matrix" instance!'.format('whitepoint', value))
            value = as_float_array(value)

        self._whitepoint = value

        self._derive_transformation_matrices()

    @property
    def whitepoint_name(self):
        """
        Getter and setter property for the whitepoint_name.

        Parameters
        ----------
        value : str
            Value to set the whitepoint_name with.

        Returns
        -------
        str
            *RGB* colourspace whitepoint name.
        """

        return self._whitepoint_name

    @whitepoint_name.setter
    def whitepoint_name(self, value):
        """
        Setter for the **self.whitepoint_name** property.
        """

        if value is not None:
            attest(
                is_string(value),
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'whitepoint_name', value))

        self._whitepoint_name = value

    @property
    def matrix_RGB_to_XYZ(self):
        """
        Getter and setter property for the transformation matrix from
        colourspace to *CIE XYZ* tristimulus values.

        Parameters
        ----------
        value : array_like
            Transformation matrix from colourspace to *CIE XYZ* tristimulus
            values.

        Returns
        -------
        array_like
            Transformation matrix from colourspace to *CIE XYZ* tristimulus
            values.
        """

        if self._use_derived_matrix_RGB_to_XYZ:
            return self._derived_matrix_RGB_to_XYZ
        else:
            return self._matrix_RGB_to_XYZ

    @matrix_RGB_to_XYZ.setter
    def matrix_RGB_to_XYZ(self, value):
        """
        Setter for the **self.matrix_RGB_to_XYZ** property.
        """

        if value is not None:
            value = as_float_array(value)

        self._matrix_RGB_to_XYZ = value

    @property
    def matrix_XYZ_to_RGB(self):
        """
        Getter and setter property for the transformation matrix from *CIE XYZ*
        tristimulus values to colourspace.

        Parameters
        ----------
        value : array_like
            Transformation matrix from *CIE XYZ* tristimulus values to
            colourspace.

        Returns
        -------
        array_like
            Transformation matrix from *CIE XYZ* tristimulus values to
            colourspace.
        """

        if self._use_derived_matrix_XYZ_to_RGB:
            return self._derived_matrix_XYZ_to_RGB
        else:
            return self._matrix_XYZ_to_RGB

    @matrix_XYZ_to_RGB.setter
    def matrix_XYZ_to_RGB(self, value):
        """
        Setter for the **self.matrix_XYZ_to_RGB** property.
        """

        if value is not None:
            value = as_float_array(value)

        self._matrix_XYZ_to_RGB = value

    @property
    def cctf_encoding(self):
        """
        Getter and setter property for the encoding colour component transfer
        function (Encoding CCTF) / opto-electronic transfer function
        (OETF / OECF).

        Parameters
        ----------
        value : callable
            Encoding colour component transfer function (Encoding CCTF) /
            opto-electronic transfer function (OETF / OECF).

        Returns
        -------
        callable
            Encoding colour component transfer function (Encoding CCTF) /
            opto-electronic transfer function (OETF / OECF).
        """

        return self._cctf_encoding

    @cctf_encoding.setter
    def cctf_encoding(self, value):
        """
        Setter for the **self.cctf_encoding** property.
        """

        if value is not None:
            attest(
                hasattr(value, '__call__'),
                '"{0}" attribute: "{1}" is not callable!'.format(
                    'cctf_encoding', value))

        self._cctf_encoding = value

    @property
    def cctf_decoding(self):
        """
        Getter and setter property for the decoding colour component transfer
        function (Decoding CCTF) / electro-optical transfer function
        (EOTF / EOCF).

        Parameters
        ----------
        value : callable
            Decoding colour component transfer function (Decoding CCTF) /
            electro-optical transfer function (EOTF / EOCF).

        Returns
        -------
        callable
            Decoding colour component transfer function (Decoding CCTF) /
            electro-optical transfer function (EOTF / EOCF).
        """

        return self._cctf_decoding

    @cctf_decoding.setter
    def cctf_decoding(self, value):
        """
        Setter for the **self.cctf_decoding** property.
        """

        if value is not None:
            attest(
                hasattr(value, '__call__'),
                '"{0}" attribute: "{1}" is not callable!'.format(
                    'cctf_decoding', value))

        self._cctf_decoding = value

    @property
    def use_derived_matrix_RGB_to_XYZ(self):
        """
        Getter and setter property for whether to use the instantiation time
        normalised primary matrix or to use a computed derived normalised
        primary matrix.

        Parameters
        ----------
        value : bool
            Whether to use the instantiation time normalised primary matrix or
            to use a computed derived normalised primary matrix.

        Returns
        -------
        bool
            Whether to use the instantiation time normalised primary matrix or
            to use a computed derived normalised primary matrix.
        """

        return self._use_derived_matrix_RGB_to_XYZ

    @use_derived_matrix_RGB_to_XYZ.setter
    def use_derived_matrix_RGB_to_XYZ(self, value):
        """
        Setter for the **self.use_derived_matrix_RGB_to_XYZ** property.
        """

        # TODO: Revisit for potential behaviour / type checking.
        self._use_derived_matrix_RGB_to_XYZ = value

    @property
    def use_derived_matrix_XYZ_to_RGB(self):
        """
        Getter and setter property for Whether to use the instantiation time
        inverse normalised primary matrix or to use a computed derived inverse
        normalised primary matrix.

        Parameters
        ----------
        value : bool
            Whether to use the instantiation time inverse normalised primary
            matrix or to use a computed derived inverse normalised primary
            matrix.

        Returns
        -------
        bool
            Whether to use the instantiation time inverse normalised primary
            matrix or to use a computed derived inverse normalised primary
            matrix.
        """

        return self._use_derived_matrix_XYZ_to_RGB

    @use_derived_matrix_XYZ_to_RGB.setter
    def use_derived_matrix_XYZ_to_RGB(self, value):
        """
        Setter for the **self.use_derived_matrix_XYZ_to_RGB** property.
        """

        # TODO: Revisit for potential behaviour / type checking.
        self._use_derived_matrix_XYZ_to_RGB = value

    def __str__(self):
        """
        Returns a formatted string representation of the *RGB* colourspace.

        Returns
        -------
        str
            Formatted string representation.

        Examples
        --------
        >>> p = np.array(
        ...     [0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
        >>> whitepoint = np.array([0.32168, 0.33767])
        >>> matrix_RGB_to_XYZ = np.identity(3)
        >>> matrix_XYZ_to_RGB = np.identity(3)
        >>> cctf_encoding = lambda x: x
        >>> cctf_decoding = lambda x: x
        >>> print(RGB_Colourspace('RGB Colourspace', p, whitepoint, 'ACES',
        ...                       matrix_RGB_to_XYZ, matrix_XYZ_to_RGB,
        ...                       cctf_encoding, cctf_decoding))
        ... # doctest: +ELLIPSIS
        RGB Colourspace
        ---------------
        <BLANKLINE>
        Primaries          : [[  7.34700000e-01   2.65300000e-01]
                              [  0.00000000e+00   1.00000000e+00]
                              [  1.00000000e-04  -7.70000000e-02]]
        Whitepoint         : [ 0.32168  0.33767]
        Whitepoint Name    : ACES
        Encoding CCTF      : <function <lambda> at 0x...>
        Decoding CCTF      : <function <lambda> at 0x...>
        NPM                : [[ 1.  0.  0.]
                              [ 0.  1.  0.]
                              [ 0.  0.  1.]]
        NPM -1             : [[ 1.  0.  0.]
                              [ 0.  1.  0.]
                              [ 0.  0.  1.]]
        Derived NPM        : \
[[  9.5255239...e-01   0.0000000...e+00   9.3678631...e-05]
                             \
 [  3.4396645...e-01   7.2816609...e-01  -7.2132546...e-02]
                             \
 [  0.0000000...e+00   0.0000000...e+00   1.0088251...e+00]]
        Derived NPM -1     : \
[[  1.0498110...e+00   0.0000000...e+00  -9.7484540...e-05]
                             \
 [ -4.9590302...e-01   1.3733130...e+00   9.8240036...e-02]
                             \
 [  0.0000000...e+00   0.0000000...e+00   9.9125201...e-01]]
        Use Derived NPM    : False
        Use Derived NPM -1 : False
        """

        def _indent_array(a):
            """
            Indents given array string representation.
            """

            return str(a).replace(' [', ' ' * 22 + '[')

        return ('{0}\n'
                '{1}\n\n'
                'Primaries          : {2}\n'
                'Whitepoint         : {3}\n'
                'Whitepoint Name    : {4}\n'
                'Encoding CCTF      : {5}\n'
                'Decoding CCTF      : {6}\n'
                'NPM                : {7}\n'
                'NPM -1             : {8}\n'
                'Derived NPM        : {9}\n'
                'Derived NPM -1     : {10}\n'
                'Use Derived NPM    : {11}\n'
                'Use Derived NPM -1 : {12}').format(
                    self.name,
                    '-' * len(self.name),
                    _indent_array(self.primaries),
                    self.whitepoint,
                    self.whitepoint_name,
                    self.cctf_encoding,
                    self.cctf_decoding,
                    _indent_array(self._matrix_RGB_to_XYZ),
                    _indent_array(self._matrix_XYZ_to_RGB),
                    _indent_array(self._derived_matrix_RGB_to_XYZ),
                    _indent_array(self._derived_matrix_XYZ_to_RGB),
                    self.use_derived_matrix_RGB_to_XYZ,
                    self.use_derived_matrix_XYZ_to_RGB,
                )

    def __repr__(self):
        """
        Returns an (almost) evaluable string representation of the *RGB*
        colourspace.

        Returns
        -------
        str
            (Almost) evaluable string representation.

        Examples
        --------
        >>> p = np.array(
        ...     [0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
        >>> whitepoint = np.array([0.32168, 0.33767])
        >>> matrix_RGB_to_XYZ = np.identity(3)
        >>> matrix_XYZ_to_RGB = np.identity(3)
        >>> cctf_encoding = lambda x: x
        >>> cctf_decoding = lambda x: x
        >>> RGB_Colourspace('RGB Colourspace', p, whitepoint, 'ACES',
        ...                 matrix_RGB_to_XYZ, matrix_XYZ_to_RGB,
        ...                 cctf_encoding, cctf_decoding)
        ... # doctest: +ELLIPSIS
        RGB_Colourspace(RGB Colourspace,
                        [[  7.34700000e-01,   2.65300000e-01],
                         [  0.00000000e+00,   1.00000000e+00],
                         [  1.00000000e-04,  -7.70000000e-02]],
                        [ 0.32168,  0.33767],
                        ACES,
                        [[ 1.,  0.,  0.],
                         [ 0.,  1.,  0.],
                         [ 0.,  0.,  1.]],
                        [[ 1.,  0.,  0.],
                         [ 0.,  1.,  0.],
                         [ 0.,  0.,  1.]],
                        <function <lambda> at 0x...>,
                        <function <lambda> at 0x...>,
                        False,
                        False)
        """

        def _indent_array(a):
            """
            Indents given array evaluable string representation.
            """

            representation = repr(a).replace(' [', '{0}['.format(' ' * 11))
            representation = representation.replace('array(', ' ' * 16)
            return representation.replace(')', '')

        return ('RGB_Colourspace({0},\n'
                '{2},\n'
                '{3},\n'
                '{1}{4},\n'
                '{5},\n'
                '{6},\n'
                '{1}{7},\n'
                '{1}{8},\n'
                '{1}{9},\n'
                '{1}{10})').format(
                    self.name,
                    ' ' * 16,
                    _indent_array(self.primaries),
                    _indent_array(self.whitepoint),
                    self.whitepoint_name,
                    _indent_array(self.matrix_RGB_to_XYZ),
                    _indent_array(self.matrix_XYZ_to_RGB),
                    self.cctf_encoding,
                    self.cctf_decoding,
                    self.use_derived_matrix_RGB_to_XYZ,
                    self.use_derived_matrix_XYZ_to_RGB,
                )

    def _derive_transformation_matrices(self):
        """
        Computes the derived transformations matrices, the normalised primary
        matrix and its inverse.
        """

        if hasattr(self, '_primaries') and hasattr(self, '_whitepoint'):
            if self._primaries is not None and self._whitepoint is not None:
                npm = normalised_primary_matrix(self._primaries,
                                                self._whitepoint)

                self._derived_matrix_RGB_to_XYZ = npm
                self._derived_matrix_XYZ_to_RGB = np.linalg.inv(npm)

    def use_derived_transformation_matrices(self, usage=True):
        """
        Enables or disables usage of both derived transformations matrices,
        the normalised primary matrix and its inverse in subsequent
        computations.

        Parameters
        ----------
        usage : bool, optional
            Whether to use the derived transformations matrices.

        Returns
        -------
        bool
            Definition success.
        """

        self.use_derived_matrix_RGB_to_XYZ = usage
        self.use_derived_matrix_XYZ_to_RGB = usage

        return True

    def chromatically_adapt(self,
                            whitepoint,
                            whitepoint_name=None,
                            chromatic_adaptation_transform='CAT02'):
        """
        Chromatically adapts the *RGB* colourspace *primaries* :math:`xy`
        chromaticity coordinates from *RGB* colourspace whitepoint to reference
        ``whitepoint``.

        Parameters
        ----------
        whitepoint : array_like
            Reference illuminant / whitepoint :math:`xy` chromaticity
            coordinates.
        whitepoint_name : str, optional
            Reference illuminant / whitepoint name.
        chromatic_adaptation_transform : str, optional
            **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
            'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02 Brill 2008', 'CAT16',
            'Bianco 2010', 'Bianco PC 2010'}**,
            *Chromatic adaptation* transform.

        Returns
        -------
        Chromatically adapted *RGB* colourspace.

        Examples
        --------
        >>> p = np.array(
        ...     [0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
        >>> w_t = np.array([0.32168, 0.33767])
        >>> w_r = np.array([0.31270, 0.32900])
        >>> colourspace = RGB_Colourspace('RGB Colourspace', p, w_t, 'D65')
        >>> print(colourspace.chromatically_adapt(w_r, 'D50', 'Bradford'))
        ... # doctest: +ELLIPSIS
        RGB Colourspace - Chromatically Adapted to D50
        ----------------------------------------------
        <BLANKLINE>
        Primaries          : [[ 0.73485524  0.26422533]
                              [-0.00617091  1.01131496]
                              [ 0.01596756 -0.0642355 ]]
        Whitepoint         : [ 0.3127  0.329 ]
        Whitepoint Name    : D50
        Encoding CCTF      : None
        Decoding CCTF      : None
        NPM                : None
        NPM -1             : None
        Derived NPM        : [[ 0.93827985 -0.00445145  0.01662752]
                              [ 0.33736889  0.72952157 -0.06689046]
                              [ 0.00117395 -0.00371071  1.09159451]]
        Derived NPM -1     : [[ 1.06349549  0.00640891 -0.01580679]
                              [-0.49207413  1.36822341  0.09133709]
                              [-0.00281646  0.00464417  0.91641857]]
        Use Derived NPM    : True
        Use Derived NPM -1 : True
        """

        colourspace = self.copy()

        colourspace.primaries = chromatically_adapted_primaries(
            colourspace.primaries, colourspace.whitepoint, whitepoint,
            chromatic_adaptation_transform)
        colourspace.whitepoint = whitepoint
        colourspace.whitepoint_name = whitepoint_name

        colourspace._matrix_RGB_to_XYZ = None
        colourspace._matrix_XYZ_to_RGB = None
        colourspace._derive_transformation_matrices()
        colourspace.use_derived_transformation_matrices()

        colourspace.name = '{0} - Chromatically Adapted to {1}'.format(
            colourspace.name, whitepoint
            if whitepoint_name is None else whitepoint_name)

        return colourspace

    def copy(self):
        """
        Returns a copy of the *RGB* colourspace.

        Returns
        -------
        RGB_Colourspace
            *RGB* colourspace copy.
        """

        return deepcopy(self)


def XYZ_to_RGB(XYZ,
               illuminant_XYZ,
               illuminant_RGB,
               matrix_XYZ_to_RGB,
               chromatic_adaptation_transform='CAT02',
               cctf_encoding=None):
    """
    Converts from *CIE XYZ* tristimulus values to *RGB* colourspace array.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant_XYZ : array_like
        *CIE xy* chromaticity coordinates or *CIE xyY* colourspace array of the
        *illuminant* for the input *CIE XYZ* tristimulus values.
    illuminant_RGB : array_like
        *CIE xy* chromaticity coordinates or *CIE xyY* colourspace array of the
        *illuminant* for the output *RGB* colourspace array.
    matrix_XYZ_to_RGB : array_like
        Matrix converting the *CIE XYZ* tristimulus values to *RGB* colourspace
        array, i.e. the inverse *Normalised Primary Matrix* (NPM).
    chromatic_adaptation_transform : str, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02 Brill 2008', 'CAT16',
        'Bianco 2010', 'Bianco PC 2010', None}**,
        *Chromatic adaptation* transform, if *None* no chromatic adaptation is
        performed.
    cctf_encoding : object, optional
        Encoding colour component transfer function (Encoding CCTF) or
        opto-electronic transfer function (OETF / OECF).

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----

    +--------------------+-----------------------+---------------+
    | **Domain**         | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``XYZ``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+
    | ``illuminant_XYZ`` | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+
    | ``illuminant_RGB`` | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    +--------------------+-----------------------+---------------+
    | **Range**          | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``RGB``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    Examples
    --------
    >>> XYZ = np.array([0.21638819, 0.12570000, 0.03847493])
    >>> illuminant_XYZ = np.array([0.34570, 0.35850])
    >>> illuminant_RGB = np.array([0.31270, 0.32900])
    >>> chromatic_adaptation_transform = 'Bradford'
    >>> matrix_XYZ_to_RGB = np.array(
    ...     [[3.24062548, -1.53720797, -0.49862860],
    ...      [-0.96893071, 1.87575606, 0.04151752],
    ...      [0.05571012, -0.20402105, 1.05699594]]
    ... )
    >>> XYZ_to_RGB(XYZ, illuminant_XYZ, illuminant_RGB, matrix_XYZ_to_RGB,
    ...            chromatic_adaptation_transform)  # doctest: +ELLIPSIS
    array([ 0.4559557...,  0.0303970...,  0.0408724...])
    """

    XYZ = to_domain_1(XYZ)

    if chromatic_adaptation_transform is not None:
        M_CAT = matrix_chromatic_adaptation_VonKries(
            xyY_to_XYZ(xy_to_xyY(illuminant_XYZ)),
            xyY_to_XYZ(xy_to_xyY(illuminant_RGB)),
            transform=chromatic_adaptation_transform)

        XYZ = vector_dot(M_CAT, XYZ)

    RGB = vector_dot(matrix_XYZ_to_RGB, XYZ)

    if cctf_encoding is not None:
        with domain_range_scale('ignore'):
            RGB = cctf_encoding(RGB)

    return from_range_1(RGB)


def RGB_to_XYZ(RGB,
               illuminant_RGB,
               illuminant_XYZ,
               matrix_RGB_to_XYZ,
               chromatic_adaptation_transform='CAT02',
               cctf_decoding=None):
    """
    Converts given *RGB* colourspace array to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    illuminant_RGB : array_like
        *CIE xy* chromaticity coordinates or *CIE xyY* colourspace array of the
        *illuminant* for the input *RGB* colourspace array.
    illuminant_XYZ : array_like
        *CIE xy* chromaticity coordinates or *CIE xyY* colourspace array of the
        *illuminant* for the output *CIE XYZ* tristimulus values.
    matrix_RGB_to_XYZ : array_like
        Matrix converting the *RGB* colourspace array to *CIE XYZ* tristimulus
        values, i.e. the *Normalised Primary Matrix* (NPM).
    chromatic_adaptation_transform : str, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02 Brill 2008', 'CAT16',
        'Bianco 2010', 'Bianco PC 2010', None}**,
        *Chromatic adaptation* transform, if *None* no chromatic adaptation is
        performed.
    cctf_decoding : object, optional
        Decoding colour component transfer function (Decoding CCTF) or
        electro-optical transfer function (EOTF / EOCF).

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +--------------------+-----------------------+---------------+
    | **Domain**         | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``RGB``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+
    | ``illuminant_XYZ`` | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+
    | ``illuminant_RGB`` | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    +--------------------+-----------------------+---------------+
    | **Range**          | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``XYZ``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    Examples
    --------
    >>> RGB = np.array([0.45595571, 0.03039702, 0.04087245])
    >>> illuminant_RGB = np.array([0.31270, 0.32900])
    >>> illuminant_XYZ = np.array([0.34570, 0.35850])
    >>> chromatic_adaptation_transform = 'Bradford'
    >>> matrix_RGB_to_XYZ = np.array(
    ...     [[0.41240000, 0.35760000, 0.18050000],
    ...      [0.21260000, 0.71520000, 0.07220000],
    ...      [0.01930000, 0.11920000, 0.95050000]]
    ... )
    >>> RGB_to_XYZ(RGB, illuminant_RGB, illuminant_XYZ, matrix_RGB_to_XYZ,
    ...            chromatic_adaptation_transform)  # doctest: +ELLIPSIS
    array([ 0.2163881...,  0.1257    ,  0.0384749...])
    """

    RGB = to_domain_1(RGB)

    if cctf_decoding is not None:
        with domain_range_scale('ignore'):
            RGB = cctf_decoding(RGB)

    XYZ = vector_dot(matrix_RGB_to_XYZ, RGB)

    if chromatic_adaptation_transform is not None:
        M_CAT = matrix_chromatic_adaptation_VonKries(
            xyY_to_XYZ(xy_to_xyY(illuminant_RGB)),
            xyY_to_XYZ(xy_to_xyY(illuminant_XYZ)),
            transform=chromatic_adaptation_transform)

        XYZ = vector_dot(M_CAT, XYZ)

    return from_range_1(XYZ)


def matrix_RGB_to_RGB(input_colourspace,
                      output_colourspace,
                      chromatic_adaptation_transform='CAT02'):
    """
    Computes the matrix :math:`M` converting from given input *RGB*
    colourspace to output *RGB* colourspace using given *chromatic
    adaptation* method.

    Parameters
    ----------
    input_colourspace : RGB_Colourspace
        *RGB* input colourspace.
    output_colourspace : RGB_Colourspace
        *RGB* output colourspace.
    chromatic_adaptation_transform : str, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02 Brill 2008', 'CAT16',
        'Bianco 2010', 'Bianco PC 2010', None}**,
        *Chromatic adaptation* transform, if *None* no chromatic adaptation is
        performed.

    Returns
    -------
    ndarray
        Conversion matrix :math:`M`.

    Examples
    --------
    >>> from colour.models import (
    ...    RGB_COLOURSPACE_sRGB, RGB_COLOURSPACE_PROPHOTO_RGB)
    >>> matrix_RGB_to_RGB(RGB_COLOURSPACE_sRGB, RGB_COLOURSPACE_PROPHOTO_RGB)
    ... # doctest: +ELLIPSIS
    array([[ 0.5288241...,  0.3340609...,  0.1373616...],
           [ 0.0975294...,  0.8790074...,  0.0233981...],
           [ 0.0163599...,  0.1066124...,  0.8772485...]])
    """

    M = input_colourspace.matrix_RGB_to_XYZ

    if chromatic_adaptation_transform is not None:
        M_CAT = matrix_chromatic_adaptation_VonKries(
            xy_to_XYZ(input_colourspace.whitepoint),
            xy_to_XYZ(output_colourspace.whitepoint),
            chromatic_adaptation_transform)

        M = matrix_dot(M_CAT, input_colourspace.matrix_RGB_to_XYZ)

    M = matrix_dot(output_colourspace.matrix_XYZ_to_RGB, M)

    return M


def RGB_to_RGB(RGB,
               input_colourspace,
               output_colourspace,
               chromatic_adaptation_transform='CAT02',
               apply_cctf_decoding=False,
               apply_cctf_encoding=False,
               **kwargs):
    """
    Converts given *RGB* colourspace array from given input *RGB* colourspace
    to output *RGB* colourspace using given *chromatic adaptation* method.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    input_colourspace : RGB_Colourspace
        *RGB* input colourspace.
    output_colourspace : RGB_Colourspace
        *RGB* output colourspace.
    chromatic_adaptation_transform : str, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02 Brill 2008', 'CAT16',
        'Bianco 2010', 'Bianco PC 2010', None}**,
        *Chromatic adaptation* transform, if *None* no chromatic adaptation is
        performed.
    apply_cctf_decoding : bool, optional
        Apply input colourspace decoding colour component transfer function /
        electro-optical transfer function.
    apply_cctf_encoding : bool, optional
        Apply output colourspace encoding colour component transfer function /
        opto-electronic transfer function.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments for the colour component transfer functions.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----

    +--------------------+-----------------------+---------------+
    | **Domain**         | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``RGB``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    +--------------------+-----------------------+---------------+
    | **Range**          | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``RGB``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    Examples
    --------
    >>> from colour.models import (
    ...     RGB_COLOURSPACE_sRGB, RGB_COLOURSPACE_PROPHOTO_RGB)
    >>> RGB = np.array([0.45595571, 0.03039702, 0.04087245])
    >>> RGB_to_RGB(RGB, RGB_COLOURSPACE_sRGB, RGB_COLOURSPACE_PROPHOTO_RGB)
    ... # doctest: +ELLIPSIS
    array([ 0.2568891...,  0.0721446...,  0.0465553...])
    """

    RGB = to_domain_1(RGB)

    if apply_cctf_decoding:
        with domain_range_scale('ignore'):
            RGB = input_colourspace.cctf_decoding(
                RGB, **filter_kwargs(input_colourspace.cctf_decoding,
                                     **kwargs))

    M = matrix_RGB_to_RGB(input_colourspace, output_colourspace,
                          chromatic_adaptation_transform)

    RGB = vector_dot(M, RGB)

    if apply_cctf_encoding:
        with domain_range_scale('ignore'):
            RGB = output_colourspace.cctf_encoding(
                RGB, **filter_kwargs(output_colourspace.cctf_encoding,
                                     **kwargs))

    return from_range_1(RGB)
