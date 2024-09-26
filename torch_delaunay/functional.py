import torch_delaunay._C as _C
from torch import Tensor


def shull2d(points: Tensor) -> Tensor:
    return _C.shull2d(points)


def circumcenter2d(p0: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    return _C.circumcenter2d(p0, p1, p2)


def circumradius2d(p0: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    return _C.circumradius2d(p0, p1, p2)
