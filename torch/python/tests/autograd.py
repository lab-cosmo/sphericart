import unittest

import sphericart_torch

import torch

torch.manual_seed(0)


class TestAutograd(unittest.TestCase):
    def test_autograd_cartesian(self):
        xyz = 6 * torch.randn(100, 3, dtype=torch.float64, requires_grad=True)

        def compute(xyz):
            return sphericart_torch.spherical_harmonics(
                l_max=20, xyz=xyz, normalize=False
            )

        self.assertTrue(torch.autograd.gradcheck(compute, xyz, fast_mode=True))

    def test_autograd_normalized(self):
        xyz = 6 * torch.randn(100, 3, dtype=torch.float64, requires_grad=True)

        def compute(xyz):
            return sphericart_torch.spherical_harmonics(
                l_max=20, xyz=xyz, normalize=True
            )

        self.assertTrue(torch.autograd.gradcheck(compute, xyz, fast_mode=True))


if __name__ == "__main__":
    unittest.main()
