import torch

from .. import SolidHarmonics as RawSolidHarmonics
from .. import SphericalHarmonics as RawSphericalHarmonics
from sphericart.metatensor.spherical_harmonics import (
    _check_xyz_tensor_map,
    _wrap_into_tensor_map,
)


try:
    import metatensor.torch
    from metatensor.torch import Labels, TensorMap
except ImportError as e:
    raise ImportError(
        "the `sphericart.torch.metatensor` module requires "
        "`metatensor-torch` to be installed"
    ) from e


class SphericalHarmonics:

    def __init__(self, l_max: int):
        self.l_max = l_max
        self.raw_calculator = RawSphericalHarmonics(l_max)

        # precompute some labels
        self.precomputed_keys = Labels(
            names=["o3_lambda"],
            values=torch.arange(l_max + 1).reshape(-1, 1),
        )
        self.precomputed_mu_components = [
            Labels(
                names=["o3_mu"],
                values=torch.arange(-l, l + 1).reshape(-1, 1),
            )
            for l in range(l_max + 1)  # noqa E741
        ]
        self.precomputed_xyz_components = Labels(
            names=["xyz"],
            values=torch.arange(3).reshape(-1, 1),
        )
        self.precomputed_xyz_2_components = Labels(
            names=["xyz_2"],
            values=torch.arange(3).reshape(-1, 1),
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
            self.precomputed_xyz_2_components,
            self.precomputed_properties,
            metatensor_module=metatensor.torch,
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
            self.precomputed_xyz_2_components,
            self.precomputed_properties,
            sh_gradients,
            metatensor_module=metatensor.torch,
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
            self.precomputed_xyz_2_components,
            sh_gradients,
            sh_hessians,
            metatensor_module=metatensor.torch,
        )


class SolidHarmonics:

    def __init__(self, l_max: int):
        self.l_max = l_max
        self.raw_calculator = RawSolidHarmonics(l_max)

        # precompute some labels
        self.precomputed_keys = Labels(
            names=["o3_lambda"],
            values=torch.arange(l_max + 1).reshape(-1, 1),
        )
        self.precomputed_mu_components = [
            Labels(
                names=["o3_mu"],
                values=torch.arange(-l, l + 1).reshape(-1, 1),
            )
            for l in range(l_max + 1)  # noqa E741
        ]
        self.precomputed_xyz_components = Labels(
            names=["xyz"],
            values=torch.arange(3).reshape(-1, 1),
        )
        self.precomputed_xyz_2_components = Labels(
            names=["xyz_2"],
            values=torch.arange(3).reshape(-1, 1),
        )
        self.precomputed_properties = Labels.single()

    def compute(self, xyz: torch.Tensor) -> TensorMap:
        _check_xyz_tensor_map(xyz)
        sh_values = self.raw_calculator.compute(xyz.block().values.squeeze(-1))
        return _wrap_into_tensor_map(
            sh_values,
            self.precomputed_keys,
            xyz.block().samples,
            self.precomputed_mu_components,
            self.precomputed_xyz_components,
            self.precomputed_xyz_2_components,
            self.precomputed_properties,
            metatensor_module=metatensor.torch,
        )

    def compute_with_gradients(self, xyz: torch.Tensor) -> TensorMap:
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
            self.precomputed_xyz_2_components,
            self.precomputed_properties,
            sh_gradients,
            metatensor_module=metatensor.torch,
        )

    def compute_with_hessians(self, xyz: torch.Tensor) -> TensorMap:
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
            self.precomputed_xyz_2_components,
            self.precomputed_properties,
            sh_gradients,
            sh_hessians,
            metatensor_module=metatensor.torch,
        )
