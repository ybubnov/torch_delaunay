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

#include <ATen/TensorAccessor.h>
#include <torch/torch.h>

#include <torch_delaunay/predicate.h>


namespace torch_delaunay {


torch::Tensor
orient2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);


template <typename scalar_t>
scalar_t
orient2d_kernel(
    const at::TensorAccessor<scalar_t, 1>& p0,
    const at::TensorAccessor<scalar_t, 1>& p1,
    const at::TensorAccessor<scalar_t, 1>& p2
)
{
    const auto d0_x = p0[0] - p2[0];
    const auto d0_y = p0[1] - p2[1];
    const auto d1_x = p1[0] - p2[0];
    const auto d1_y = p1[1] - p2[1];
    return d0_x * d1_y - d0_y * d1_x;
}


torch::Tensor
incircle2d(
    const torch::Tensor& p0,
    const torch::Tensor& p1,
    const torch::Tensor& p2,
    const torch::Tensor& q
);


torch::Tensor
incircle2d(const torch::Tensor& points, const torch::Tensor& q);


template <typename scalar_t>
scalar_t
incircle2d_kernel(
    const at::TensorAccessor<scalar_t, 1>& p0,
    const at::TensorAccessor<scalar_t, 1>& p1,
    const at::TensorAccessor<scalar_t, 1>& p2,
    const at::TensorAccessor<scalar_t, 1>& q
)
{
    const auto d0_x = p0[0] - q[0];
    const auto d1_x = p1[0] - q[0];
    const auto d2_x = p2[0] - q[0];
    const auto d0_y = p0[1] - q[1];
    const auto d1_y = p1[1] - q[1];
    const auto d2_y = p2[1] - q[1];

    const auto d0_s = d0_x * d0_x + d0_y * d0_y;
    const auto d1_s = d1_x * d1_x + d1_y * d1_y;
    const auto d2_s = d2_x * d2_x + d2_y * d2_y;

    return (
        d0_x * (d1_y * d2_s - d1_s * d2_y) - d0_y * (d1_x * d2_s - d1_s * d2_x)
        + d0_s * (d1_x * d2_y - d1_y * d2_x)
    );
}


} // namespace torch_delaunay
