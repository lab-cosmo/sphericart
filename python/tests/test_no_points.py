import numpy as np
import pytest

import sphericart


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("l_max", [0, 3, 7, 10, 20, 50])
def test_no_points(l_max, normalized):
    xyz = np.empty((0, 3))

    if normalized:
        calculator = sphericart.SphericalHarmonics(l_max)
    else:
        calculator = sphericart.SolidHarmonics(l_max)
    sph_sphericart = calculator.compute(xyz)
    assert sph_sphericart.shape == (0, l_max * l_max + 2 * l_max + 1)
