from typing import List, Optional

import numpy as np
from metatensor import Labels, TensorBlock, TensorMap

from ..spherical_harmonics import SolidHarmonics as RawSolidHarmonics
from ..spherical_harmonics import SphericalHarmonics as RawSphericalHarmonics


class SphericalHarmonics:

    def __init__(self, l_max: int):
        self.l_max = l_max
        self.raw_calculator = RawSphericalHarmonics(l_max)

        # precompute some labels
        self.precomputed_keys = Labels(
            names=["o3_lambda"],
            values=np.arange(l_max + 1).reshape(-1, 1),
        )
        self.precomputed_mu_components = [
            Labels(
                names=["o3_mu"],
                values=np.arange(-l, l + 1).reshape(-1, 1),
            )
            for l in range(l_max + 1)  # noqa E741
        ]
        self.precomputed_xyz_components = Labels(
            names=["xyz"],
            values=np.arange(2).reshape(-1, 1),
        )
        self.precomputed_xyz_1_components = Labels(
            names=["xyz_1"],
            values=np.arange(2).reshape(-1, 1),
        )
        self.precomputed_xyz_2_components = Labels(
            names=["xyz_2"],
            values=np.arange(2).reshape(-1, 1),
        )
        self.precomputed_properties = Labels.single()

    def compute(self, xyz: TensorMap) -> TensorMap:
        _check_xyz_tensor_map(xyz)
        sh_values = self.raw_calculator.compute(xyz.block().values.squeeze(-1))
        return _wrap_into_tensor_map(
            sh_values,
            self.precomputed_keys,
            xyz.block().samples,
            self.precomputed_mu_components,
            self.precomputed_xyz_components,
            self.precomputed_xyz_1_components,
            self.precomputed_xyz_2_components,
            self.precomputed_properties,
        )

    def compute_with_gradients(self, xyz: TensorMap) -> TensorMap:
        _check_xyz_tensor_map(xyz)
        sh_values, sh_gradients = self.raw_calculator.compute_with_gradients(
            xyz.block().values.squeeze(-1)
        )
        return _wrap_into_tensor_map(
            sh_values,
            self.precomputed_keys,
            xyz.block().samples,
            self.precomputed_mu_components,
            self.precomputed_xyz_components,
            self.precomputed_xyz_1_components,
            self.precomputed_xyz_2_components,
            self.precomputed_properties,
            sh_gradients,
        )

    def compute_with_hessians(self, xyz: TensorMap) -> TensorMap:
        _check_xyz_tensor_map(xyz)
        sh_values, sh_gradients, sh_hessians = (
            self.raw_calculator.compute_with_hessians(xyz.block().values.squeeze(-1))
        )
        return _wrap_into_tensor_map(
            sh_values,
            self.precomputed_keys,
            xyz.block().samples,
            self.precomputed_mu_components,
            self.precomputed_properties,
            self.precomputed_xyz_components,
            self.precomputed_xyz_1_components,
            self.precomputed_xyz_2_components,
            sh_gradients,
            sh_hessians,
        )


class SolidHarmonics:

    def __init__(self, l_max: int):
        self.l_max = l_max
        self.raw_calculator = RawSolidHarmonics(l_max)

        # precompute some labels
        self.precomputed_keys = Labels(
            names=["o3_lambda"],
            values=np.arange(l_max + 1).reshape(-1, 1),
        )
        self.precomputed_mu_components = [
            Labels(
                names=["o3_mu"],
                values=np.arange(-l, l + 1).reshape(-1, 1),
            )
            for l in range(l_max + 1)  # noqa E741
        ]
        self.precomputed_xyz_components = Labels(
            names=["xyz"],
            values=np.arange(2).reshape(-1, 1),
        )
        self.precomputed_xyz_1_components = Labels(
            names=["xyz_1"],
            values=np.arange(2).reshape(-1, 1),
        )
        self.precomputed_xyz_2_components = Labels(
            names=["xyz_2"],
            values=np.arange(2).reshape(-1, 1),
        )
        self.precomputed_properties = Labels.single()

    def compute(self, xyz: np.ndarray) -> TensorMap:
        _check_xyz_tensor_map(xyz)
        sh_values = self.raw_calculator.compute(xyz.block().values.squeeze(-1))
        return _wrap_into_tensor_map(
            sh_values,
            self.precomputed_keys,
            xyz.block().samples,
            self.precomputed_mu_components,
            self.precomputed_xyz_components,
            self.precomputed_xyz_1_components,
            self.precomputed_xyz_2_components,
            self.precomputed_properties,
        )

    def compute_with_gradients(self, xyz: np.ndarray) -> TensorMap:
        _check_xyz_tensor_map(xyz)
        sh_values, sh_gradients = self.raw_calculator.compute_with_gradients(
            xyz.block().values.squeeze(-1)
        )
        return _wrap_into_tensor_map(
            sh_values,
            self.precomputed_keys,
            xyz.block().samples,
            self.precomputed_mu_components,
            self.precomputed_xyz_components,
            self.precomputed_xyz_1_components,
            self.precomputed_xyz_2_components,
            self.precomputed_properties,
            sh_gradients,
        )

    def compute_with_hessians(self, xyz: np.ndarray) -> TensorMap:
        _check_xyz_tensor_map(xyz)
        sh_values, sh_gradients, sh_hessians = (
            self.raw_calculator.compute_with_hessians(xyz.block().values.squeeze(-1))
        )
        return _wrap_into_tensor_map(
            sh_values,
            self.precomputed_keys,
            xyz.block().samples,
            self.precomputed_mu_components,
            self.precomputed_xyz_components,
            self.precomputed_xyz_1_components,
            self.precomputed_xyz_2_components,
            self.precomputed_properties,
            sh_gradients,
            sh_hessians,
        )


def _check_xyz_tensor_map(xyz: TensorMap):
    if len(xyz.blocks()) != 1:
        raise ValueError("`xyz` should have only one block")
    if len(xyz.block().components) != 1:
        raise ValueError("`xyz` should have only one component")
    if xyz.block().components[0].names != ["xyz"]:
        raise ValueError("`xyz` should have only one component named 'xyz'")
    if xyz.block().components[0].values.shape[0] != 3:
        raise ValueError("`xyz` should have 3 Cartesian coordinates")
    if xyz.block().properties.values.shape[0] != 1:
        raise ValueError("`xyz` should have only one property")


def _wrap_into_tensor_map(
    sh_values: np.ndarray,
    keys: Labels,
    samples: Labels,
    components: List[Labels],
    xyz_components: Labels,
    xyz_1_components: Labels,
    xyz_2_components: Labels,
    properties: Labels,
    sh_gradients: Optional[np.ndarray] = None,
    sh_hessians: Optional[np.ndarray] = None,
) -> TensorMap:

    # infer l_max
    l_max = len(components) - 1

    blocks = []
    for l in range(l_max + 1):  # noqa E741
        l_start = l**2
        l_end = (l + 1) ** 2
        sh_values_block = TensorBlock(
            values=sh_values[:, l_start:l_end, None],
            samples=samples,
            components=[components[l]],
            properties=properties,
        )
        if sh_gradients is not None:
            sh_gradients_block = TensorBlock(
                values=sh_gradients[:, :, l_start:l_end, None],
                samples=samples,
                components=[components[l], xyz_components],
                properties=properties,
            )
            if sh_hessians is not None:
                sh_hessians_block = TensorBlock(
                    values=sh_hessians[:, :, :, l_start:l_end, None],
                    samples=samples,
                    components=[
                        components[l],
                        xyz_1_components,
                        xyz_2_components,
                    ],
                    properties=properties,
                )
                sh_gradients_block.add_gradient("positions", sh_hessians_block)
            sh_values_block.add_gradient("positions", sh_gradients_block)

        blocks.append(sh_values_block)

    return TensorMap(keys=keys, blocks=blocks)
