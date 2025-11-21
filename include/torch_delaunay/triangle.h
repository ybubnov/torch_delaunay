// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

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
std::tuple<double, double>
circumcircle2d_kernel(
    const at::TensorAccessor<scalar_t, 1> p0,
    const at::TensorAccessor<scalar_t, 1> p1,
    const at::TensorAccessor<scalar_t, 1> p2
)
{
    const auto bx = p1[0] - p0[0], by = p1[1] - p0[1];
    const auto cx = p2[0] - p0[0], cy = p2[1] - p0[1];

    const double d = static_cast<double>(2 * (bx * cy - by * cx));

    const auto b_norm = bx * bx + by * by;
    const auto c_norm = cx * cx + cy * cy;

    const double ux = (cy * b_norm - by * c_norm) / d;
    const double uy = (bx * c_norm - cx * b_norm) / d;

    return std::forward_as_tuple(ux, uy);
}


template <typename scalar_t>
torch::Tensor
circumcenter2d_kernel(
    const at::TensorAccessor<scalar_t, 1> p0,
    const at::TensorAccessor<scalar_t, 1> p1,
    const at::TensorAccessor<scalar_t, 1> p2
)
{
    const auto [ux, uy] = circumcircle2d_kernel<scalar_t>(p0, p1, p2);
    return torch::tensor({ux + p0[0], uy + p0[1]}, torch::dtype(torch::kFloat64));
}


template <typename scalar_t>
double
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
