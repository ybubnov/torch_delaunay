#pragma once

#include <torch/torch.h>


namespace torchgis {
namespace en {

torch::Tensor triangle_orient2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);

torch::Tensor triangle_circumcenter2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);

torch::Tensor triangle_circumradius2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);

torch::Tensor dist(const torch::Tensor& points, const torch::Tensor& q);

torch::Tensor shull2d(const torch::Tensor& points);


} // namespace en
} // namespace torchgis
