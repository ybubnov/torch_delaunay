#pragma once

#include <torch/torch.h>


namespace torchgis {
namespace en {

torch::Tensor orient2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);

torch::Tensor incircle2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2, const torch::Tensor& q);

torch::Tensor circumcenter2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);

torch::Tensor circumradius2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);

torch::Tensor dist(const torch::Tensor& points, const torch::Tensor& q);

torch::Tensor shull2d(const torch::Tensor& points);

torch::Tensor lawson_flip(const torch::Tensor& triangles, const torch::Tensor& points);


} // namespace en
} // namespace torchgis
