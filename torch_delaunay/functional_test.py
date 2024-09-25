import torch

from torch_delaunay.functional import shull2d


def test_shull2d() -> None:
    points = torch.randn((1000, 2), dtype=torch.float64)
    simplices = shull2d(points)

    assert simplices.shape[1] == 3
    assert simplices.shape[0] > 0
