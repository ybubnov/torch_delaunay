from typing import Tuple

import torch
import torchgis._C as _C


def lawson_flip(
    triangles: torch.Tensor,
    points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _C.lawson_flip(triangles, points)


def shull2d(points: torch.Tensor) -> torch.Tensor:
    return _C.shull2d(points)
