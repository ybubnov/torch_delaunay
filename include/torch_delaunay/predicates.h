#pragma once

#include <torch/torch.h>


namespace torch_delaunay {


torch::Tensor
orient2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);


torch::Tensor
incircle2d(
    const torch::Tensor& p0,
    const torch::Tensor& p1,
    const torch::Tensor& p2,
    const torch::Tensor& q
);


bool
incircle2d(const torch::Tensor& points, const torch::Tensor& q);


} // namespace torch_delaunay
