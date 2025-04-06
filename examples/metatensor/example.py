import numpy as np
from metatensor import Labels, TensorBlock, TensorMap

import sphericart
import sphericart.metatensor


l_max = 15
n_samples = 100

xyz = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=np.random.rand(n_samples, 3, 1),
                samples=Labels(
                    names=["sample"],
                    values=np.arange(n_samples).reshape(-1, 1),
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

calculator = sphericart.metatensor.SphericalHarmonics(l_max)

spherical_harmonics = calculator.compute(xyz)
# for each block, the samples are the same as those of the `xyz` input

for single_l in range(l_max + 1):
    spherical_single_l = spherical_harmonics.block({"o3_lambda": single_l})

    # check values against pure sphericart
    assert np.allclose(
        spherical_single_l.values.squeeze(-1),
        sphericart.SphericalHarmonics(single_l).compute(
            xyz.block().values.squeeze(-1)
        )[:, single_l**2 : (single_l + 1) ** 2],
    )

# further example: obtaining gradients of l = 2 spherical harmonics
spherical_harmonics = calculator.compute_with_gradients(xyz)
l_2_gradients = spherical_harmonics.block({"o3_lambda": 2}).gradient("positions")

# further example: obtaining Hessians of l = 2 spherical harmonics
spherical_harmonics = calculator.compute_with_hessians(xyz)
l_2_hessians = spherical_harmonics.block(
    {"o3_lambda": 2}
).gradient("positions").gradient("positions")
