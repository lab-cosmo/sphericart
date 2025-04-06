from typing import List, Optional

import torch

from . import SolidHarmonics as RawSolidHarmonics
from . import SphericalHarmonics as RawSphericalHarmonics


try:
    from metatensor.torch import Labels, TensorBlock, TensorMap
except ImportError as e:
    raise ImportError(
        "the `sphericart.torch.metatensor` module requires "
        "`metatensor-torch` to be installed"
    ) from e


class SphericalHarmonics:
    """
    ``metatensor``-based wrapper around the
    :py:meth:`sphericart.torch.SphericalHarmonics` class.

    See :py:class:`sphericart.metatensor.SphericalHarmonics` for more details.
    ``backward_second_derivatives`` has the same meaning as in
    :py:class:`sphericart.torch.SphericalHarmonics`.
    """

    def __init__(
        self,
        l_max: int,
        backward_second_derivatives: bool = False,
    ):
        self.l_max = l_max
        self.raw_calculator = RawSphericalHarmonics(l_max, backward_second_derivatives)

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
        """
        See :py:meth:`sphericart.metatensor.SphericalHarmonics.compute`.
        """
        _check_xyz_tensor_map(xyz)
        device = xyz.device
        if self.precomputed_keys.device != device:
            self._send_precomputed_labels_to_device(device)

        sh_values = self.raw_calculator.compute(xyz.block().values.squeeze(-1))
        return _wrap_into_tensor_map(
            sh_values,
            self.precomputed_keys,
            xyz.block().samples,
            self.precomputed_mu_components,
            self.precomputed_xyz_components,
            self.precomputed_xyz_2_components,
            self.precomputed_properties,
        )

    def compute_with_gradients(self, xyz: TensorMap) -> TensorMap:
        """
        See :py:meth:`sphericart.metatensor.SphericalHarmonics.compute_with_gradients`.
        """
        _check_xyz_tensor_map(xyz)
        device = xyz.device
        if self.precomputed_keys.device != device:
            self._send_precomputed_labels_to_device(device)

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
        )

    def compute_with_hessians(self, xyz: TensorMap) -> TensorMap:
        """
        See :py:meth:`sphericart.metatensor.SphericalHarmonics.compute_with_hessians`.
        """
        _check_xyz_tensor_map(xyz)
        device = xyz.device
        if self.precomputed_keys.device != device:
            self._send_precomputed_labels_to_device(device)

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
        )

    def _send_precomputed_labels_to_device(self, device):
        self.precomputed_keys = self.precomputed_keys.to(device)
        self.precomputed_mu_components = [
            comp.to(device) for comp in self.precomputed_mu_components
        ]
        self.precomputed_xyz_components = self.precomputed_xyz_components.to(device)
        self.precomputed_xyz_2_components = self.precomputed_xyz_2_components.to(device)
        self.precomputed_properties = self.precomputed_properties.to(device)


class SolidHarmonics:
    """
    ``metatensor``-based wrapper around the
    :py:meth:`sphericart.torch.SolidHarmonics` class.

    See :py:class:`sphericart.metatensor.SphericalHarmonics` for more details.
    ``backward_second_derivatives`` has the same meaning as in
    :py:class:`sphericart.torch.SphericalHarmonics`.
    """

    def __init__(
        self,
        l_max: int,
        backward_second_derivatives: bool = False,
    ):
        self.l_max = l_max
        self.raw_calculator = RawSolidHarmonics(l_max, backward_second_derivatives)

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
        """
        See :py:meth:`sphericart.metatensor.SphericalHarmonics.compute`.
        """
        _check_xyz_tensor_map(xyz)
        device = xyz.device
        if self.precomputed_keys.device != device:
            self._send_precomputed_labels_to_device(device)

        sh_values = self.raw_calculator.compute(xyz.block().values.squeeze(-1))
        return _wrap_into_tensor_map(
            sh_values,
            self.precomputed_keys,
            xyz.block().samples,
            self.precomputed_mu_components,
            self.precomputed_xyz_components,
            self.precomputed_xyz_2_components,
            self.precomputed_properties,
        )

    def compute_with_gradients(self, xyz: TensorMap) -> TensorMap:
        """
        See :py:meth:`sphericart.metatensor.SphericalHarmonics.compute_with_gradients`.
        """
        _check_xyz_tensor_map(xyz)
        device = xyz.device
        if self.precomputed_keys.device != device:
            self._send_precomputed_labels_to_device(device)

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
        )

    def compute_with_hessians(self, xyz: TensorMap) -> TensorMap:
        """
        See :py:meth:`sphericart.metatensor.SphericalHarmonics.compute_with_hessians`.
        """
        _check_xyz_tensor_map(xyz)
        device = xyz.device
        if self.precomputed_keys.device != device:
            self._send_precomputed_labels_to_device(device)

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
        )

    def _send_precomputed_labels_to_device(self, device):
        self.precomputed_keys = self.precomputed_keys.to(device)
        self.precomputed_mu_components = [
            comp.to(device) for comp in self.precomputed_mu_components
        ]
        self.precomputed_xyz_components = self.precomputed_xyz_components.to(device)
        self.precomputed_xyz_2_components = self.precomputed_xyz_2_components.to(device)
        self.precomputed_properties = self.precomputed_properties.to(device)


def _check_xyz_tensor_map(xyz: TensorMap):
    blocks = xyz.blocks()
    if len(blocks) != 1:
        raise ValueError("`xyz` should have only one block")

    block = blocks[0]
    components = block.components
    if len(components) != 1:
        raise ValueError("`xyz` should have only one component")
    if components[0].names != ["xyz"]:
        raise ValueError("`xyz` should have only one component named 'xyz'")

    values_shape = block.values.shape
    if values_shape[1] != 3:
        raise ValueError("`xyz` should have 3 Cartesian coordinates")
    if values_shape[2] != 1:
        raise ValueError("`xyz` should have only one property")


def _wrap_into_tensor_map(
    sh_values: torch.Tensor,
    keys: Labels,
    samples: Labels,
    components: List[Labels],
    xyz_components: Labels,
    xyz_2_components: Labels,
    properties: Labels,
    sh_gradients: Optional[torch.Tensor] = None,
    sh_hessians: Optional[torch.Tensor] = None,
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
                components=[xyz_components, components[l]],
                properties=properties,
            )
            if sh_hessians is not None:
                sh_hessians_block = TensorBlock(
                    values=sh_hessians[:, :, :, l_start:l_end, None],
                    samples=samples,
                    components=[
                        xyz_2_components,
                        xyz_components,
                        components[l],
                    ],
                    properties=properties,
                )
                sh_gradients_block.add_gradient("positions", sh_hessians_block)
            sh_values_block.add_gradient("positions", sh_gradients_block)

        blocks.append(sh_values_block)

    return TensorMap(keys=keys, blocks=blocks)
