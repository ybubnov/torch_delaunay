#include <torchgis/en/predicates.h>

#include <torch/linalg.h>


using namespace torch::indexing;


namespace torchgis {
namespace en {


torch::Tensor orient2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2) {
  const auto d0 = p1 - p0;
  const auto d1 = (p2 - p1).index({Slice(), torch::tensor({1, 0})});

  return (d0 * d1).diff(1).sign();
}


torch::Tensor incircle2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2, const torch::Tensor& q) {

  const auto d0 = p0 - q;
  const auto d1 = p1 - q;
  const auto d2 = p2 - q;
  const auto d = torch::stack({d0, d1, d2}, 1);

  const auto abc = d.square().sum(2);
  const auto A = torch::cat({d, abc.view({-1, 3, 1})}, -1);

  return torch::linalg::det(A).sign();
}



} // namespace en
} // namespace torchgis
