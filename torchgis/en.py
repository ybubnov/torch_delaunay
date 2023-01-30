import torch
import torchgis._en as _en


def circumradius2d(input: torch.Tensor) -> torch.Tensor:
    return _en.circumradius2d(input)


def circumcenter2d(input: torch.Tensor) -> torch.Tensor:
    return _en.circumcenter2d(input)


def dist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return _en.dist(p, q)


def shull2d(points: torch.Tensor) -> torch.Tensor:
    return _en.shull2d(points)
