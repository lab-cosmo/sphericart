import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

import sphericart.torch
import sphericart.torch.metatensor


L_MAX = 15
N_SAMPLES = 100


@pytest.fixture
def xyz():
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.rand(N_SAMPLES, 3, 1),
                samples=Labels(
                    names=["sample"],
                    values=torch.arange(N_SAMPLES).reshape(-1, 1),
                ),
                components=[
                    Labels(
                        names=["xyz"],
                        values=torch.arange(3).reshape(-1, 1),
                    )
                ],
                properties=Labels.single(),
            )
        ],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_metatensor(xyz, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    xyz = xyz.to(device)
    for l in range(L_MAX + 1):  # noqa E741
        calculator_spherical = sphericart.torch.metatensor.SphericalHarmonics(l)
        calculator_solid = sphericart.torch.metatensor.SolidHarmonics(l)

        spherical = calculator_spherical.compute(xyz)
        solid = calculator_solid.compute(xyz)

        assert spherical.keys == Labels(
            names=["o3_lambda"],
            values=torch.arange(l + 1).reshape(-1, 1),
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
                    values=torch.arange(-single_l, single_l + 1).reshape(-1, 1),
                )
            ]

            # check properties
            assert spherical_block.properties == Labels.single()

            # check values
            assert torch.allclose(
                spherical_block.values.squeeze(-1),
                sphericart.torch.SphericalHarmonics(single_l).compute(
                    xyz.block().values.squeeze(-1)
                )[:, single_l**2 : (single_l + 1) ** 2],
            )
            assert torch.allclose(
                solid_block.values.squeeze(-1),
                sphericart.torch.SolidHarmonics(l).compute(
                    xyz.block().values.squeeze(-1)
                )[:, single_l**2 : (single_l + 1) ** 2],
            )
