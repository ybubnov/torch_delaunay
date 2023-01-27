#pragma once

#include <torch/torch.h>

namespace torchgis {
namespace cartesian {


torch::Tensor circumradius2d(const torch::Tensor& input);


} // namespace cartesian
} // namespace torchgis
