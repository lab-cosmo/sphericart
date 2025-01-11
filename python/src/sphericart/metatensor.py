from typing import List, Optional

import numpy as np

from .spherical_harmonics import SolidHarmonics as RawSolidHarmonics
from .spherical_harmonics import SphericalHarmonics as RawSphericalHarmonics


try:
    from metatensor import Labels, TensorBlock, TensorMap
except ImportError as e:
    raise ImportError(
        "the `sphericart.metatensor` module requires `metatensor` to be installed"
    ) from e


class SphericalHarmonics:
    """
    ``metatensor``-based wrapper around the
    :py:meth:`sphericart.SphericalHarmonics` class.

    :param l_max: the maximum degree of the spherical harmonics to be calculated

    :return: a spherical harmonics calculator object
    """

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
            values=np.arange(3).reshape(-1, 1),
        )
        self.precomputed_xyz_2_components = Labels(
            names=["xyz_2"],
            values=np.arange(3).reshape(-1, 1),
        )
        self.precomputed_properties = Labels.single()

    def compute(self, xyz: TensorMap) -> TensorMap:
        """
        Computes the spherical harmonics for the given Cartesian coordinates, up to
        the maximum degree ``l_max`` specified during initialization.

        :param xyz: a :py:class:`metatensor.TensorMap` containing the Cartesian
            coordinates of the 3D points. This ``TensorMap`` should have only one
            ``TensorBlock``. In this ``TensorBlock``, the samples are arbitrary,
            there must be one component named ``"xyz"`` with 3 values, and one property.

        :return: The spherical harmonics and their metadata as a
            :py:class:`metatensor.TensorMap`. All ``samples`` in the output
            ``TensorMap`` will be the same as those of the ``xyz`` input.
        """
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
        )

    def compute_with_gradients(self, xyz: TensorMap) -> TensorMap:
        """
        Computes the spherical harmonics for the given Cartesian coordinates, up to
        the maximum degree ``l_max`` specified during initialization,
        together with their gradients with respect to the Cartesian coordinates.

        :param xyz: see :py:meth:`compute`

        :return: The spherical harmonics and their metadata as a
            :py:class:`metatensor.TensorMap`. Each ``TensorBlock`` in the output
            ``TensorMap`` will have a gradient block with respect to the Cartesian
            positions. All ``samples`` in the output ``TensorMap`` will be the same as
            those of the ``xyz`` input.
        """
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
        )

    def compute_with_hessians(self, xyz: TensorMap) -> TensorMap:
        """
        Computes the spherical harmonics for the given Cartesian coordinates, up to
        the maximum degree ``l_max`` specified during initialization,
        together with their gradients and Hessians with respect to the Cartesian
        coordinates.

        :param xyz: see :py:meth:`compute`

        :return: The spherical harmonics and their metadata as a
            :py:class:`metatensor.TensorMap`. Each ``TensorBlock`` in the output
            ``TensorMap`` will have a gradient block  with respect to the Cartesian
            positions, which will itself have a gradient with respect to the Cartesian
            positions. All ``samples`` in the output ``TensorMap`` will be the same as
            those of the ``xyz`` input.
        """
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
        )


class SolidHarmonics:
    """
    ``metatensor``-based wrapper around the :py:meth:`sphericart.SolidHarmonics` class.

    See :py:class:`SphericalHarmonics` for more details.
    """

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
            values=np.arange(3).reshape(-1, 1),
        )
        self.precomputed_xyz_2_components = Labels(
            names=["xyz_2"],
            values=np.arange(3).reshape(-1, 1),
        )
        self.precomputed_properties = Labels.single()

    def compute(self, xyz: TensorMap) -> TensorMap:
        """
        See :py:meth:`sphericart.metatensor.SphericalHarmonics.compute`.
        """
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
        )

    def compute_with_gradients(self, xyz: TensorMap) -> TensorMap:
        """
        See :py:meth:`sphericart.metatensor.SphericalHarmonics.compute_with_gradients`.
        """
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
        )

    def compute_with_hessians(self, xyz: TensorMap) -> TensorMap:
        """
        See :py:meth:`sphericart.metatensor.SphericalHarmonics.compute_with_hessians`.
        """
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
