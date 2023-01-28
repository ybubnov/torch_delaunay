#pragma once

#include <torch/torch.h>


namespace torchgis {
namespace cartesian {

torch::Tensor circumcenter2d(const torch::Tensor& input);

torch::Tensor circumradius2d(const torch::Tensor& input);


} // namespace cartesian
} // namespace torchgis
