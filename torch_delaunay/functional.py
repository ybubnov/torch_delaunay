import torch
import torch_delaunay._C as _C


def shull2d(points: torch.Tensor) -> torch.Tensor:
    return _C.shull2d(points)
