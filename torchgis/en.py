import torch
import torchgis._c as _c


def circumradius2d(input: torch.Tensor) -> torch.Tensor:
    return _c.circumradius2d(input)


def circumcenter2d(input: torch.Tensor) -> torch.Tensor:
    return _c.circumcenter2d(input)


def shull2d(points: torch.Tensor) -> torch.Tensor:
    return _c.shull2d(points)
