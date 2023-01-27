#include <torchgis/cartesian.h>

#include <ATen/Functions.h>
#include <torch/linalg.h>

using namespace torch::indexing;


namespace torchgis {
namespace cartesian {

torch::Tensor circumradius2d(const torch::Tensor& input) {
  // Dimensions:
  //  0 - triangles
  //  1 - point (A, B, C)
  //  2 - x and y coordinate

  auto B = input.index({Slice(), 1}) - input.index({Slice(), 0});
  auto C = input.index({Slice(), 2}) - input.index({Slice(), 0});

  std::initializer_list<at::indexing::TensorIndex> x = {Slice(), 0};
  std::initializer_list<at::indexing::TensorIndex> y = {Slice(), 1};

  auto D = 2 * (B.index(x) * C.index(y) - B.index(y) * C.index(x));

  auto B_norm = B.square().sum({1});
  auto C_norm = C.square().sum({1});

  auto Ux = (C.index(y) * B_norm - B.index(y) * C_norm) / D;
  auto Uy = (B.index(x) * C_norm - C.index(x) * B_norm) / D;

  return (Ux.square() + Uy.square()).sqrt();
}

} // cartesian
} // torchgis
