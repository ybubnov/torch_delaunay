#include <torchgis/en/predicates.h>

#include <torch/linalg.h>


using namespace torch::indexing;


namespace torchgis {
namespace en {


torch::Tensor orient2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2
) {
  const auto dx = p0 - p2;
  const auto dy = p1 - p2;

  const auto A = torch::stack({dx, dy}, 1);
  return -torch::linalg::det(A).sign();
}


torch::Tensor incircle2d(
  const torch::Tensor& p0,
  const torch::Tensor& p1,
  const torch::Tensor& p2,
  const torch::Tensor& q
) {

  const auto d0 = p0 - q;
  const auto d1 = p1 - q;
  const auto d2 = p2 - q;
  const auto d = torch::stack({d0, d1, d2}, 1);

  const auto abc = d.square().sum(2);
  const auto A = torch::cat({d, abc.view({-1, 3, 1})}, -1);

  return torch::linalg::det(A).sign();
}


bool all_clockwise2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2
) {
  return torchgis::en::orient2d(p0, p1, p2).lt(0.0).all().item<bool>();
}


bool all_counterclockwise2d(
  const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2
) {
  return torchgis::en::orient2d(p0, p1, p2).gt(0.0).all().item<bool>();
}


} // namespace en
} // namespace torchgis
