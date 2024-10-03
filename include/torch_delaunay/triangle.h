#pragma once

#include <torch/torch.h>


namespace torch_delaunay {


torch::Tensor
circumcenter2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);


torch::Tensor
circumradius2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);


} // namespace torch_delaunay
