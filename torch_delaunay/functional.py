# Copyright (C) 2024, Yakau Bubnou
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch_delaunay._C as _C
from torch import Tensor


def shull2d(points: Tensor) -> Tensor:
    """Computes Delaunay tessellation for 2-dimensional coordinates.

    Args:
        input: A list of 2D coordinates where each coordinate is represented as a tuple (x, y),
            where x is the longitude and y is the latitude.

    Shape:
        - Input: :math:`(*, 2)`, where 2 comprises x and y coordinates.
        - Output: :math:`(N, 3)`, where :math:`N` is a number of simplices in the output
            tessellation, and 3 represents vertices of a simplex.

    Examples:

    >>> import torch
    >>> from torch_delaunay.functional import shull2d
    >>> points = torch.rand((100, 2), dtype=torch.float64)
    >>> simplices = shull2d(points)
    """
    return _C.shull2d(points)


def circumcenter2d(p0: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    return _C.circumcenter2d(p0, p1, p2)


def circumradius2d(p0: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    return _C.circumradius2d(p0, p1, p2)
