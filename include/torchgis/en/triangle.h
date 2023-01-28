#pragma once

#include <torch/torch.h>


namespace torchgis {
namespace en {


torch::Tensor circumcenter2d(const torch::Tensor& input);

torch::Tensor circumradius2d(const torch::Tensor& input);

torch::Tensor shull2d(const torch::Tensor& points);


} // namespace en
} // namespace torchgis
