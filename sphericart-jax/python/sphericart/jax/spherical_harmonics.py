import jax

from .sph import sph


def spherical_harmonics(xyz: jax.Array, l_max: int):
    """Computes the spherical harmonics and their derivatives within
    the JAX framework.

    The definition of the real spherical harmonics is consistent with the
    Wikipedia spherical harmonics page.

    Note that the ``l_max`` argument (position 1 in the signature) should be tagged as
    static when jit-ing the function:

    >>> import jax
    >>> import sphericart.jax
    >>> jitted_sph_fn = jax.jit(sphericart.jax.spherical_harmonics, static_argnums=1)

    :param xyz: single vector or set of vectors in 3D. All dimensions are optional
        except for the last. Shape ``[..., 3]``.
    :param l_max: the maximum degree of the spherical harmonics to be calculated
        (included)

    :return: Spherical harmonics expansion of ``xyz``. Shape ``[..., (l_max+1)**2]``.
        The last dimension is organized in lexicographic order.
        For example, if ``l_max = 2``, The last axis will correspond to
        spherical harmonics with ``(l, m) = (0, 0), (1, -1), (1, 0), (1,
        1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)``, in this order.
    """
    if xyz.shape[-1] != 3:
        raise ValueError("the last axis of xyz must have size 3")
    xyz = xyz.ravel().reshape(xyz.shape)  # make contiguous (???)
    output = sph(xyz, l_max, normalized=True)
    return output


def solid_harmonics(xyz: jax.Array, l_max: int):
    """
    Same as `spherical_harmonics`, but computes the solid harmonics instead.

    These are a non-normalized form of the real
    spherical harmonics, i.e. :math:`r^l Y^l_m`. These scaled spherical harmonics
    are polynomials in the Cartesian coordinates of the input points, and they
    are therefore less expoensive to compute.
    """
    if xyz.shape[-1] != 3:
        raise ValueError("the last axis of xyz must have size 3")
    xyz = xyz.ravel().reshape(xyz.shape)  # make contiguous (???)
    output = sph(xyz, l_max, normalized=False)
    return output
