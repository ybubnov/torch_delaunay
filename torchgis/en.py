from typing import Tuple

import torch
import torchgis._en as _en


def incircle2d(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return _en.incircle2d(p0, p1, p2, q)

def circumradius2d(input: torch.Tensor) -> torch.Tensor:
    return _en.circumradius2d(input)


def circumcenter2d(input: torch.Tensor) -> torch.Tensor:
    return _en.circumcenter2d(input)


def dist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return _en.dist(p, q)


def lawson_flip(
    triangles: torch.Tensor, points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _en.lawson_flip(triangles, points)


def shull2d(points: torch.Tensor) -> torch.Tensor:
    return _en.shull2d(points)
