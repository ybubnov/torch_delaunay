// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <ATen/TensorAccessor.h>
#include <torch/torch.h>


namespace torch_delaunay {


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
