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
            self.linear_layer = torch.nn.Linear(4, 1)

        def forward(self, positions):
            positions.requires_grad = True
            sh = self.sh_calculator.compute(positions)
            energy = torch.sum(self.linear_layer(sh))
            forces = torch.autograd.grad(
                energy,
                xyz,
                retain_graph=True,
                create_graph=True,
            )[0]
            return energy, forces

    nn = NN()
    energy, forces = nn(xyz)
    loss = (target - energy) ** 2 + torch.sum((d_target - forces) ** 2)
    print(xyz.requires_grad)
    loss.backward()
