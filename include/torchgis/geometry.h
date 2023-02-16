#pragma once

#include <torch/torch.h>


namespace torchgis {

struct LineTensor;

//
// (0, 0) (1, 2) (4, 5)
// (3, 2) (9, 9)
//
// points: [
//   0: [0, 0],
//   1: [1, 2],
//   2: [4, 5],
//   3: [3, 2],
//   4: [9, 9],
// ]
//
// view: [
//   0: [0, 3],
//   1: [3, 5],
// ]
//
template <typename T>
std::tuple<torch::Tensor, torch::Tensor> linestrings(
  std::initializer_list<T> points, const torch::TensorOptions& options
) {
  // ...
}


}
