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

#pragma once

#include <cmath>

#include <torch/torch.h>


namespace torch_delaunay {


torch::Tensor
circumcenter2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);


torch::Tensor
circumcenter2d(const torch::Tensor& points);


torch::Tensor
circumradius2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);


torch::Tensor
circumradius2d(const torch::Tensor& points);


template <typename scalar_t>
std::tuple<scalar_t, scalar_t>
circumcircle2d_kernel(
    const at::TensorAccessor<scalar_t, 1> p0,
    const at::TensorAccessor<scalar_t, 1> p1,
    const at::TensorAccessor<scalar_t, 1> p2
)
{
    const auto ax = p0[0], ay = p0[1];
    const auto bx = p1[0] - p0[0], by = p1[1] - p0[1];
    const auto cx = p2[0] - p0[0], cy = p2[1] - p0[1];

    const auto d = 2 * (bx * cy - by * cx);

    const auto b_norm = bx * bx + by * by;
    const auto c_norm = cx * cx + cy * cy;

    const auto ux = (cy * b_norm - by * c_norm) / d;
    const auto uy = (bx * c_norm - cx * b_norm) / d;

    return std::forward_as_tuple(ux, uy);
}


template <typename scalar_t>
scalar_t
circumradius2d_kernel(
    const at::TensorAccessor<scalar_t, 1> p0,
    const at::TensorAccessor<scalar_t, 1> p1,
    const at::TensorAccessor<scalar_t, 1> p2
)
{
    const auto [ux, uy] = circumcircle2d_kernel<scalar_t>(p0, p1, p2);
    return std::sqrt(ux * ux + uy * uy);
}

} // namespace torch_delaunay
