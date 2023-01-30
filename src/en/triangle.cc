#include <torchgis/en/triangle.h>

#include <torch/linalg.h>


using namespace torch::indexing;


namespace torchgis {
namespace en {


torch::Tensor triangle_orient2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2) {
  auto d0 = p1 - p0;
  auto d1 = (p2 - p1).view({Slice(), {1, 0}});

  return (d0 * d1).diff({1}).less(0.0);
}


torch::Tensor triangle_circumcenter2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2) {
  TORCH_CHECK(p0.dim() == 2, "triangle_circumcenter2d only supports 2D tensors, got: ", p0.dim(), "D");
  TORCH_CHECK(p1.dim() == 2, "triangle_circumcenter2d only supports 2D tensors, got: ", p1.dim(), "D");
  TORCH_CHECK(p2.dim() == 2, "triangle_circumcenter2d only supports 2D tensors, got: ", p2.dim(), "D");

  auto a = p0;
  auto b = -p0 + p1;
  auto c = -p0 + p2;

  std::initializer_list<at::indexing::TensorIndex> x = {Slice(), 0};
  std::initializer_list<at::indexing::TensorIndex> y = {Slice(), 1};

  auto d = 2 * (b.index(x) * c.index(y) - b.index(y) * c.index(x));

  auto b_norm = b.square().sum({1});
  auto c_norm = c.square().sum({1});

  auto ux = (c.index(y) * b_norm - b.index(y) * c_norm) / d;
  auto uy = (b.index(x) * c_norm - c.index(x) * b_norm) / d;

  return at::column_stack({ux, uy}) + a;
}


torch::Tensor triangle_circumradius2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2) {
  return torchgis::en::triangle_circumcenter2d(p0, p1, p2).square().sum({1}).sqrt();
}


torch::Tensor dist(const torch::Tensor& p, const torch::Tensor& q) {
  TORCH_CHECK(p.dim() == 2, "dist only supports 2D tensors, got: ", p.dim(), "D");

  return torch::linalg::norm(p - q, /*ord=*/2, /*dim=*/1, false, c10::nullopt);
}


torch::Tensor shull2d(const torch::Tensor& points) {
  TORCH_CHECK(points.dim() == 2, "shull2d only supports 2D tensors, got: ", points.dim(), "D");

  // Indices of the seed triangle.
  torch::Tensor i0, i1, i2;

  // Choose seed points close to a centroid of the point cloud.
  {
    torch::Tensor min, max;
    std::tie(min, max) = points.aminmax(0);

    auto centroid = (max + min) / 2;
    auto dists = torchgis::en::dist(points, centroid);

    torch::Tensor values, indices;
    std::tie(values, indices) = at::topk(dists, 2, /*dim=*/-1, /*largest=*/false, /*sorted=1*/true);

    i0 = indices[0];
    i1 = indices[1];
  }

  // Find the third point such that forms the smallest circumcircle with i0 and i1.
  {
    auto p0 = points[i0].unsqueeze(0);
    auto p1 = points[i1].unsqueeze(0);

    auto radiuses = torchgis::en::triangle_circumradius2d(points, p0, p1);

    torch::Tensor values, indices;
    std::tie(values, indices) = at::topk(radiuses, 1, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);
    i2 = indices[0];
  }


  return at::cat({points[i0], points[i1], points[i2]});
}


} // en
} // torchgis
