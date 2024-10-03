#pragma once

#include <torch/torch.h>


namespace torch_delaunay {


torch::Tensor
shull2d(const torch::Tensor& points);


} // namespace torch_delaunay
