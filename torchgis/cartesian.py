import torch
import torchgis._c as _c


def circumradius2d(tensor: torch.Tensor):
    return _c.circumradius2d(tensor)
