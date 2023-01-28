#include <torchgis/cartesian.h>

#include <ATen/Functions.h>


using namespace torch::indexing;


namespace torchgis {
namespace cartesian {


torch::Tensor circumcenter2d(const torch::Tensor& input) {
  TORCH_CHECK(input.dim() == 3, "circumcenter2d only supports 3D tensors, got: ", input.dim(), "D");

  auto a = input.index({Slice(), 0});
  auto b = input.index({Slice(), 1}) - a;
  auto c = input.index({Slice(), 2}) - a;

  std::initializer_list<at::indexing::TensorIndex> x = {Slice(), 0};
  std::initializer_list<at::indexing::TensorIndex> y = {Slice(), 1};

  auto d = 2 * (b.index(x) * c.index(y) - b.index(y) * c.index(x));

  auto b_norm = b.square().sum({1});
  auto c_norm = c.square().sum({1});

  auto ux = (c.index(y) * b_norm - b.index(y) * c_norm) / d;
  auto uy = (b.index(x) * c_norm - c.index(x) * b_norm) / d;

  return at::column_stack({ux, uy}) + a;
}


torch::Tensor circumradius2d(const torch::Tensor& input) {
  TORCH_CHECK(input.dim() == 3, "circumradius2d only supports 3D tensors, got: ", input.dim(), "D");

  return circumcenter2d(input).square().sum({1}).sqrt();
}


} // cartesian
} // torchgis
