from .sph import sph


def spherical_harmonics(xyz, l_max, normalized=False):
    """Computes the Spherical harmonics and their derivatives within
    the JAX framework.

    This function supports ``jit``, ``vmap``, and up to two rounds of forward and/or
    backward automatic differentiation (``grad``, ``jacfwd``, ``jacrev``, ``hessian``,
    ...). For the moment, it does not support ``pmap``.

    Note that the ``l_max`` and ``normalized`` arguments (positions 1 and 2 in the
    signature) should be tagged as static when jit-ing the function:

    >>> import jax
    >>> import sphericart.jax
    >>> sph_fn_jit = jax.jit(sphericart.jax.spherical_harmonics, static_argnums=(1, 2))

    Parameters
    ----------
    xyz : jax array [..., 3]
        single vector or set of vectors in 3D. All dimensions are optional except for
        the last
    l_max : int
        maximum order of the spherical harmonics (included)
    normalized : bool
        whether the function computes Cartesian solid harmonics (``normalized=False``,
        default) or normalized spherical harmonicsi (``normalized=True``)

    Returns
    -------
    jax array [..., (l_max+1)**2]
        Spherical harmonics expansion of `xyz`
    """
    if xyz.shape[-1] != 3:
        raise ValueError("the last axis of xyz must have size 3")
    xyz = xyz.ravel().reshape(xyz.shape)  # make contiguous (???)
    output = sph(xyz, l_max, normalized)
    return output
