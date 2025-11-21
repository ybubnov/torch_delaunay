# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Yakau Bubnou
# SPDX-FileType: SOURCE

from typing import Optional
from typing import Union

from torch import Tensor

import torch_delaunay._C as _C


def shull2d(points: Tensor, eps: Optional[Union[float, int]] = None) -> Tensor:
    """Computes Delaunay tessellation for 2-dimensional coordinates.

    Note:
        This implementation automatically deletes coplanar simplices, and therefore could
        return an empty tensor in the result.

    Args:
        points: A list of 2D coordinates where each coordinate is represented as a tuple (x, y).
        eps: If specified, filters out simplices with a circumscribed circle radius smaller than
            the given value.

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
    return _C.shull2d(points, eps)


def circumcenter2d(p0: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    """Return a coordinate of the circumscribed circle center of a triangle.

    Args:
        p0: A list of 2D coordinates, representing the first vertex of a triangle.
        p1: A list of 2D coordinates, representing the second vertex of a triangle.
        p2: A list of 2D coordinates, representing the third vertex of a triangle.

    Shape:
        - Input: :math:`(*, 2)`, where 2 comprises x and y coordinates.
        - Output: :math:`(*, 2)`.

    Examples:

    >>> import torch
    >>> from torch_delaunay.functional import circumcenter2d
    >>> points = torch.rand((100, 2), dtype=torch.float64)
    >>> output = circumcenter2d(points)
    """
    return _C.circumcenter2d(p0, p1, p2)


def circumradius2d(p0: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    """Return a coordinate of the circumscribed circle radius of a triangle.

    Args:
        p0: A list of 2D coordinates, representing the first vertex of a triangle.
        p1: A list of 2D coordinates, representing the second vertex of a triangle.
        p2: A list of 2D coordinates, representing the third vertex of a triangle.

    Shape:
        - Input: :math:`(*, 2)`, where 2 comprises x and y coordinates.
        - Output: :math:`(*, 2)`.

    Examples:

    >>> import torch
    >>> from torch_delaunay.functional import circumradius2d
    >>> points = torch.rand((100, 2), dtype=torch.float64)
    >>> output = circumradius2d(points)
    """
    return _C.circumradius2d(p0, p1, p2)
