#pragma once

#include <torch/torch.h>


namespace torchgis {
namespace en {


torch::Tensor orient2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);

torch::Tensor incircle2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2, const torch::Tensor& q);


bool all_clockwise2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);


bool all_counterclockwise2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);

} // namespace en
} // namespace torchgis
