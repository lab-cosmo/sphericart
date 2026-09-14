What sphericart computes
========================

In the most basic form, spherical harmonics are defined as complex-valued functions of 
the polar coordinates :math:`(\theta,\phi)`

.. math ::
    Y^m_l(\theta,\phi) \propto P^m_l(\cos \theta) e^{\mathrm{i} m \phi}

where :math:`P^m_l(x)` are the associated Legendre polynomials.
There are multiple conventions for choosing normalization and phases, and it is 
possible to reformulate the spherical harmonics in a real-valued form, which leads
to even further ambiguity in the definitions. 

Within `sphericart`, we compute only real-valued spherical harmonics and we express
them as a function of the full Cartesian coordinates of a point in three dimensions.
These correspond to the real spherical harmonics as defined in the corresponding 
`Wikipedia article <https://en.wikipedia.org/wiki/Spherical_harmonics>`_, which we
refer to as :math:`Y^m_l`. 
If you need complex spherical harmonics, or use a different convention for normalization
and storage order it is usually simple - if tedious and inefficient - to perform the
conversion manually, see :doc:`spherical-complex` for a simple example.


We also offer the possibility to compute "solid" harmonics, which are given by
:math:`\tilde{Y}^m_l = r^l\,{Y}_l^m`. Since these can be expressed as homogeneous
polynomials of the Cartesian coordinates :math:`(x,y,z)`, as opposed to
:math:`(x/r,y/r,z/r)`, they are less computationally expensive to evaluate.
Besides being slightly faster, they can also provide a more natural scaling if 
used together with a radial expansion, and we recommend using them unless you
need the normalized version.

The formulas used to compute the solid harmonics (and, with few modifications,
also for the spherical harmonics) are:

.. math ::
    \tilde{Y}_l^m(x, y, z) = r^l\,{Y}_l^m(x, y, z) = F_l^{|m|} Q_l^{|m|}(z, r) \times
    \begin{cases}
      s_{|m|}(x, y) & \text{if $m < 0$}\\
      1/\sqrt{2} & \text{if $m = 0$}\\
      c_m(x, y) & \text{if $m > 0$}
    \end{cases}

where

.. math ::
    r =& \, \sqrt{x^2+y^2+z^2}, \quad
    r_{xy} = \sqrt{x^2+y^2}, \quad \\
    s_m =& \, r_{xy}^m \, \sin{(m \arctan(x/r,y/r))}, \quad \\
    c_m = & \, r_{xy}^m \, \cos{(m\arctan(x/r,y/r))},\label{eq:define-q}\quad \\
    Q_l^m(z,r) =&\, r^l r_{xy}^{-m} \, P_l^m(z/r), \quad \\
    F_l^m = &\, (-1)^m \sqrt{\frac{2l+1}{2\pi}\frac{(l-m)!}{(l+m)!}}.

If we neglect some constant normalization factors, these correspond to the 
`regular solid harmonics <https://en.wikipedia.org/wiki/Solid_harmonics>`_:
the functions computed by ``sphericart`` should be multiplied by 
:math:`\sqrt{4\pi/(2l+1)}` to recover the usual definition.

See also the `reference paper <https://arxiv.org/abs/2302.08381>`_ for further 
implementation details.

The :math:`\tilde{Y}^m_l(x)` are stored contiguously in memory, e.g. as
:math:`\{ (l,m)=(0,0), (1,-1), (1,0), (1,1), (2,-2), \ldots \}`. 
With zero-based indexing of the arrays, the ``(l,m)`` term is stored at 
position ``l(l+1)+m``.
