import torch
import torchgis._c as _c


def circumradius2d(tensor: torch.Tensor) -> torch.Tensor:
    return _c.circumradius2d(tensor)


def circumcenter2d(tensor: torch.Tensor) -> torch.Tensor:
    return _c.circumcenter2d(tensor)
