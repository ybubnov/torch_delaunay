/// Copyright (C) 2024, Yakau Bubnou
///
/// This program is free software: you can redistribute it and/or modify
/// it under the terms of the GNU General Public License as published by
/// the Free Software Foundation, either version 3 of the License, or
/// (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <torch_delaunay/predicate.h>

#include <ATen/ATen.h>


using namespace torch::indexing;


namespace torch_delaunay {


// Returns positive value if p0, p1, p2 are in counter-clockwise order.
//
// In other words, it returns negative value if p0, p1, p2 are in clockwise order.
torch::Tensor
orient2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    const auto dx = p0 - p2;
    const auto dy = p1 - p2;

    const auto A = torch::stack({dx, dy}, 1);
    return torch::linalg_det(A).sign();
}


// Returns a positive value if q lies inside the oriented circle p0p1p2.
//
// In other words, it returns a negative value if q lies outside the oriented circle p0p1p2.
torch::Tensor
incircle2d(
    const torch::Tensor& p0,
    const torch::Tensor& p1,
    const torch::Tensor& p2,
    const torch::Tensor& q
)
{
    const auto d0 = p0 - q;
    const auto d1 = p1 - q;
    const auto d2 = p2 - q;
    const auto d = torch::stack({d0, d1, d2}, 1);

    const auto abc = d.square().sum(2);
    const auto A = torch::cat({d, abc.view({-1, 3, 1})}, -1);

    return torch::linalg_det(A).sign();
}


torch::Tensor
incircle2d(const torch::Tensor& points, const torch::Tensor& q)
{
    const auto d = points - q;
    const auto abc = d.square().sum(-1).view({-1, 1});
    const auto A = torch::hstack({d, abc});

    return torch::linalg_det(A).sign();
}


} // namespace torch_delaunay
