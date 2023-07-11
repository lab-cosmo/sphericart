import torch

import sphericart.torch


torch.manual_seed(0)


def test_nn():
    # Make sure that we can train a NN with gradients of the target

    target = torch.randn(1)
    d_target = torch.randn(20, 3)
    xyz = 6 * torch.randn(20, 3)

    class NN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sh_calculator = sphericart.torch.SphericalHarmonics(
                l_max=1, normalized=True
            )
            self.linear_layer = torch.nn.Linear(4, 1, bias=False)

        def forward(self, positions):
            positions.requires_grad = True
            sh = self.sh_calculator.compute(positions)
            energy = torch.sum(self.linear_layer(sh))
            forces = torch.autograd.grad(
                energy,
                positions,
                retain_graph=True,
                create_graph=True,
            )[0]
            return energy, forces

    nn = NN()
    energy, forces = nn(xyz)
    loss = (target - energy) ** 2 + torch.sum((d_target - forces) ** 2)
    loss.backward()


def test_nn_consistency():
    # Make sure that, while training a NN with gradients of the target, the gradients
    # of the loss with respect to the weights are the same regardless of whether we
    # initialize the sphericart calculator with or without backward_second_derivatives

    target = torch.randn(1)
    d_target = torch.randn(20, 3)
    xyz = 6 * torch.randn(20, 3)

    class NN(torch.nn.Module):
        def __init__(self, backward_second_derivatives) -> None:
            super().__init__()
            self.sh_calculator = sphericart.torch.SphericalHarmonics(
                l_max=1,
                normalized=True,
                backward_second_derivatives=backward_second_derivatives,
            )
            self.linear_layer = torch.nn.Linear(4, 1, bias=False)
            self.linear_layer.weight = torch.nn.Parameter(
                torch.tensor([0.0, 1.0, 2.0, 3.0])
            )

        def forward(self, positions):
            positions.requires_grad = True
            sh = self.sh_calculator.compute(positions)
            energy = torch.sum(self.linear_layer(sh))
            forces = torch.autograd.grad(
                energy,
                positions,
                retain_graph=True,
                create_graph=True,
            )[0]
            return energy, forces

    nn_false = NN(backward_second_derivatives=False)
    xyz_false = xyz.detach().clone()
    energy_false, forces_false = nn_false(xyz_false)
    loss_false = (target - energy_false) ** 2 + torch.sum(
        (d_target - forces_false) ** 2
    )
    loss_false.backward()

    nn_true = NN(backward_second_derivatives=True)
    xyz_true = xyz.detach().clone()
    energy_true, forces_true = nn_true(xyz_true)
    loss_true = (target - energy_true) ** 2 + torch.sum((d_target - forces_true) ** 2)
    loss_true.backward()

    assert torch.allclose(
        nn_true.linear_layer.weight.grad, nn_false.linear_layer.weight.grad
    )
