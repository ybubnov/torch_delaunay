// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <ATen/ATen.h>

#include <torch_delaunay/predicate.h>


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
