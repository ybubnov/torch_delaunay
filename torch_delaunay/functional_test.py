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

import torch
from shapely import Point
from geopandas import GeoDataFrame
from geopandas import points_from_xy

from torch_delaunay.functional import shull2d, circumcenter2d, circumradius2d


def test_shull2d() -> None:
    points = torch.randn((1000, 2), dtype=torch.float64)
    simplices = shull2d(points)
    print("shull2d completed")

    assert (simplices >= 1000).sum() == 0, "simplices are outside of points array limit"

    assert simplices.shape[1] == 3
    assert simplices.shape[0] > 0

    vertices = GeoDataFrame(geometry=points_from_xy(points[:, 0], points[:, 1]))

    # Ensure that triangles comply with Delaunay definition.
    tri = points[simplices]

    centers = circumcenter2d(tri[:, 0], tri[:, 1], tri[:, 2]).detach().numpy()
    radii = circumradius2d(tri[:, 0], tri[:, 1], tri[:, 2]).detach().numpy()

    for center, radius in zip(centers, radii):
        circle = Point(*center).buffer(radius)
        incircle = vertices.intersection(circle)
        assert (~incircle.is_empty).sum() <= 3
