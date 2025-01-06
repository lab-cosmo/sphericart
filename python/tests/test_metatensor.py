import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

import sphericart
import sphericart.metatensor


L_MAX = 15
N_SAMPLES = 100


@pytest.fixture
def xyz():
    np.random.seed(0)
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=np.random.rand(N_SAMPLES, 3, 1),
                samples=Labels(
                    names=["sample"],
                    values=np.arange(N_SAMPLES).reshape(-1, 1),
                ),
                components=[
                    Labels(
                        names=["xyz"],
                        values=np.arange(3).reshape(-1, 1),
                    )
                ],
                properties=Labels.single(),
            )
        ],
    )


def test_metatensor(xyz):
    for l in range(L_MAX + 1):  # noqa E741
        calculator_spherical = sphericart.metatensor.SphericalHarmonics(l)
        calculator_solid = sphericart.metatensor.SolidHarmonics(l)

        spherical = calculator_spherical.compute(xyz)
        solid = calculator_solid.compute(xyz)

        assert spherical.keys == Labels(
            names=["o3_lambda"],
            values=np.arange(l + 1).reshape(-1, 1),
        )
        for single_l in range(l + 1):  # noqa E741
            spherical_block = spherical.block({"o3_lambda": single_l})
            solid_block = solid.block({"o3_lambda": single_l})

            # check samples
            assert spherical_block.samples == xyz.block().samples

            # check components
            assert spherical_block.components == [
                Labels(
                    names=["o3_mu"],
                    values=np.arange(-single_l, single_l + 1).reshape(-1, 1),
                )
            ]

            # check properties
            assert spherical_block.properties == Labels.single()

            # check values
            assert np.allclose(
                spherical_block.values.squeeze(-1),
                sphericart.SphericalHarmonics(single_l).compute(
                    xyz.block().values.squeeze(-1)
                )[:, single_l**2 : (single_l + 1) ** 2],
            )
            assert np.allclose(
                solid_block.values.squeeze(-1),
                sphericart.SolidHarmonics(l).compute(xyz.block().values.squeeze(-1))[
                    :, single_l**2 : (single_l + 1) ** 2
                ],
            )
