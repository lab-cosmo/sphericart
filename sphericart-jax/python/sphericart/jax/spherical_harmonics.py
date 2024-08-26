from .sph import sph


def spherical_harmonics(xyz, l_max):
    """Computes the Spherical harmonics and their derivatives within
    the JAX framework.

    Note that the ``l_max`` argument (position 1 in the signature) should be tagged as
    static when jit-ing the function:

    >>> import jax
    >>> import sphericart.jax
    >>> jitted_sph_fn = jax.jit(sphericart.jax.spherical_harmonics, static_argnums=1)

    Parameters
    ----------
    xyz : jax array [..., 3]
        single vector or set of vectors in 3D. All dimensions are optional except for
        the last
    l_max : int
        maximum order of the spherical harmonics (included)

    Returns
    -------
    jax array [..., (l_max+1)**2]
        Spherical harmonics expansion of `xyz`
    """
    if xyz.shape[-1] != 3:
        raise ValueError("the last axis of xyz must have size 3")
    xyz = xyz.ravel().reshape(xyz.shape)  # make contiguous (???)
    output = sph(xyz, l_max, normalized=True)
    return output


def solid_harmonics(xyz, l_max, normalized=False):
    """
    Same as `spherical_harmonics`, but computes the solid harmonics instead.
    """
    if xyz.shape[-1] != 3:
        raise ValueError("the last axis of xyz must have size 3")
    xyz = xyz.ravel().reshape(xyz.shape)  # make contiguous (???)
    output = sph(xyz, l_max, normalized=False)
    return output
