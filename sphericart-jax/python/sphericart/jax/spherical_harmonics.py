from .sph import sph


def spherical_harmonics(xyz, l_max, normalized):  # TODO: CHANGE xyz to first argument everywhere
    """Computes the Spherical harmonics and their derivatives within
    the JAX framework. See :py:class:`sphericart.SphericalHarmonics` for more details.

    Parameters
    ----------
    l_max : int
        maximum order of the spherical harmonics (included)
    normalized : bool
        should we compute cartesian (``normalized=False``) or normalized spherical harmonics
    xyz : jax array [n_sample, 3]
        set of n_sample vectors in 3D

    Returns
    -------
    jax array [n_sample, (l_max+1)**2]
        Spherical harmonics expension of `xyz`
    """
    if xyz.shape[-1] != 3: raise ValueError("the last axis of xyz must have size 3")
    xyz = xyz.ravel().reshape(xyz.shape)  # make contiguous (???)
    output = sph(xyz, l_max, normalized)
    return output
