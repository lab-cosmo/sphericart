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

Within `sphericart` we take an opinionated stance: we compute only real-valued
harmonics, we express them as a function of the full Cartesian coordinates of a 
point in three dimensions :math:`(x,y,z)` and compute by default "scaled" 
versions :math:`\tilde{Y}^m_l(x)` which correspond to polynomials of the 
Cartesian coordinates:

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

See also the `reference paper <https://arxiv.org/abs/2302.08381>`_ for further 
implementation details.

The normalized version of the spherical harmonics can also be computed by providing
the appropriate flag when creating the `sphericart` calculators, but we recommend using
the scaled versions, that are slightly faster and provide a more natural scaling 
when used together with a radial expansion.

The :math:`\tilde{Y}^m_l(x)` are stored contiguously in memory, e.g. as
:math:`\{ (l,m)=(0,0), (1,-1), (1,0), (1,1), (2,-2), \ldots \}`. 
With zero-based indexing of the arrays, the ``(l,m)`` term is stored at 
position ``l(l+1)+m``.
