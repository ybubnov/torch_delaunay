#pragma once

#include <torch/torch.h>


namespace torch_delaunay {


torch::Tensor
circumcenter2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);


torch::Tensor
circumradius2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2);


torch::Tensor
dist(const torch::Tensor& points, const torch::Tensor& q);


torch::Tensor
shull2d(const torch::Tensor& points);


torch::Tensor
lawson_flip(const torch::Tensor& triangles, const torch::Tensor& points);


} // namespace torch_delaunay
