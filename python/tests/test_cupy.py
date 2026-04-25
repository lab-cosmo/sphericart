import numpy as np
import pytest

import sphericart


cp = pytest.importorskip("cupy")


def _has_cuda_device():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cuda_device(), reason="CUDA device not available"
)


@pytest.mark.parametrize(
    "calculator_cls", [sphericart.SphericalHarmonics, sphericart.SolidHarmonics]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cupy_matches_numpy(calculator_cls, dtype):
    rng = np.random.default_rng(0)
    xyz_np = rng.normal(size=(8, 3)).astype(dtype)
    xyz_cp = cp.asarray(xyz_np)

    calculator = calculator_cls(l_max=4)

    sph_np, dsph_np, ddsph_np = calculator.compute_with_hessians(xyz_np)
    sph_cp, dsph_cp, ddsph_cp = calculator.compute_with_hessians(xyz_cp)

    assert isinstance(sph_cp, cp.ndarray)
    assert isinstance(dsph_cp, cp.ndarray)
    assert isinstance(ddsph_cp, cp.ndarray)

    if dtype == np.float32:
        rtol = 1e-5
        atol = 1e-5
    else:
        rtol = 1e-12
        atol = 1e-12

    np.testing.assert_allclose(cp.asnumpy(sph_cp), sph_np, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cp.asnumpy(dsph_cp), dsph_np, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cp.asnumpy(ddsph_cp), ddsph_np, rtol=rtol, atol=atol)
