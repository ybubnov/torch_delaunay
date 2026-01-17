/// SPDX-License-Identifier: GPL-3.0-or-later
/// SPDX-FileCopyrightText: 2025 Yakau Bubnou
/// SPDX-FileType: SOURCE

#pragma once

#include <optional>

#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>


namespace torch_delaunay {


torch::Tensor
shull2d(const torch::Tensor& points, std::optional<const c10::Scalar> eps = std::nullopt);


} // namespace torch_delaunay
