import numpy as np
import pytest

import sphericart


@pytest.mark.parametrize("normalized", [False, True], ids=["solid", "spherical"])
@pytest.mark.parametrize("l_max", [0, 3, 7, 10, 20, 50])
def test_no_points(l_max, normalized):
    xyz = np.empty((0, 3))

    if normalized:
        calculator = sphericart.SphericalHarmonics(l_max)
    else:
        calculator = sphericart.SolidHarmonics(l_max)
    sph_sphericart = calculator.compute(xyz)
    assert sph_sphericart.shape == (0, l_max * l_max + 2 * l_max + 1)


def test_double_del():
    calculator = sphericart.SphericalHarmonics(l_max=4)
    calculator.__del__()
    calculator.__del__()

    calculator = sphericart.SolidHarmonics(l_max=4)
    calculator.__del__()
    calculator.__del__()

    message = "can not use a deleted calculator"
    xyz = np.zeros((5, 3))
    with pytest.raises(ValueError, match=message):
        calculator.compute(xyz)
