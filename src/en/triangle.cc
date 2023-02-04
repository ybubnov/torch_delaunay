#include <torchgis/en/triangle.h>

#include <ATen/native/cpu/Loops.h>
#include <torch/linalg.h>


using namespace torch::indexing;


namespace torchgis {
namespace en {


torch::Tensor orient2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2) {
  const auto d0 = p1 - p0;
  const auto d1 = (p2 - p1).index({Slice(), torch::tensor({1, 0})});

  return (d0 * d1).diff(1).less(0.0);
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


torch::Tensor circumcenter2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2) {
  TORCH_CHECK(p0.dim() == 2, "circumcenter2d only supports 2D tensors, got: ", p0.dim(), "D");
  TORCH_CHECK(p1.dim() == 2, "circumcenter2d only supports 2D tensors, got: ", p1.dim(), "D");
  TORCH_CHECK(p2.dim() == 2, "circumcenter2d only supports 2D tensors, got: ", p2.dim(), "D");

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


torch::Tensor circumradius2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2) {
  return torchgis::en::circumcenter2d(p0, p1, p2).square().sum({1}).sqrt();
}


torch::Tensor dist(const torch::Tensor& p, const torch::Tensor& q) {
  TORCH_CHECK(p.dim() == 2, "dist only supports 2D tensors, got: ", p.dim(), "D");

  return torch::linalg::norm(p - q, /*ord=*/2, /*dim=*/1, false, c10::nullopt);
}


torch::Tensor lawson_flip(const torch::Tensor& triangles_, const torch::Tensor& points_) {
  torch::Tensor triangles = triangles_.contiguous();
  torch::Tensor points = points_.contiguous();

  torch::Tensor out = torch::empty(triangles.sizes(), triangles.options());

  const auto m = points.size(0)*3;

  torch::Tensor he_indices = torch::full({2, m}, -1, triangles.options());
  torch::Tensor he_values = torch::full({m}, -1, triangles.options());

  auto n = triangles.size(0);

  for (const auto i : c10::irange(n)) {
    const auto a = triangles.index({i, 0});
    const auto b = triangles.index({i, 1});
    const auto c = triangles.index({i, 2});

    he_indices.index_put_({0, i+0}, b);
    he_indices.index_put_({1, i+0}, a);
    he_indices.index_put_({0, i+1}, c);
    he_indices.index_put_({1, i+1}, b);
    he_indices.index_put_({0, i+2}, a);
    he_indices.index_put_({1, i+2}, c);

    he_values.index_put_({i+0}, i+1);
    he_values.index_put_({i+1}, i+1);
    he_values.index_put_({i+2}, i+1);
  }

  const auto mask = he_values.greater(-1);
  const auto halfedges = torch::sparse_coo_tensor(
    he_indices.index({Slice(), mask}), he_values.index({mask})
  );
  std::cout << "REV INDEX" << std::endl << halfedges.to_dense() << std::endl;

  torch::Tensor pr = triangles.index({Slice(), 0}).contiguous();
  torch::Tensor pl = triangles.index({Slice(), 1}).contiguous();
  torch::Tensor p0 = triangles.index({Slice(), 2}).contiguous();

  // TODO: implement diag for sparse coo tensor.
  auto p1_ = halfedges.index_select(0, pr).index_select(1, pl).to_dense().diag();
  std::cout << "P1'" << std::endl << p1_ << std::endl;

  // TODO: -1 is not correct value.
  torch::Tensor p1 = triangles.index({p1_ - 1, 2}).contiguous();

  const int64_t* p0_ptr = p0.data_ptr<int64_t>();
  const int64_t* pl_ptr = pl.data_ptr<int64_t>();
  const int64_t* p1_ptr = p1.data_ptr<int64_t>();

  std::cout << "P0 (points)" << std::endl << points.index({p0}) << std::endl;
  std::cout << "Pr (points)" << std::endl << points.index({pr}) << std::endl;
  std::cout << "Pl (points)" << std::endl << points.index({pl}) << std::endl;
  std::cout << "P1 (points)" << std::endl << points.index({p1}) << std::endl;

  const auto orient = torchgis::en::orient2d(
    points.index({p0}), points.index({pr}), points.index({pl})
  );
  std::cout << "ORIENT" << std::endl << orient << std::endl;

  const auto incircle = torchgis::en::incircle2d(
    points.index({p0}), points.index({pr}), points.index({pl}), points.index({p1})
  ).contiguous();
  std::cout << "IN CIRCLE" << std::endl << incircle << std::endl;

  const float* incircle_ptr = incircle.data_ptr<float>();

  for (int64_t i = 0; i < out.sizes()[0]; i++) {
    if (incircle_ptr[i] > 0) {
      out.index_put_({i, 0}, p1_ptr[i]);
      out.index_put_({i, 1}, pl_ptr[i]);
      out.index_put_({i, 2}, p0_ptr[i]);
    } else {
      out.index_put_({i}, triangles[i]);
    }
  }

  return out;
}


torch::Tensor shull2d(const torch::Tensor& points) {
  TORCH_CHECK(points.dim() == 2, "shull2d only supports 2D tensors, got: ", points.dim(), "D");

  // Indices of the seed triangle.
  torch::Tensor i0, i1, i2;

  // Choose seed points close to a centroid of the point cloud.
  {
    torch::Tensor min, max;
    std::tie(min, max) = points.aminmax(0);

    const auto centroid = (max + min) / 2;
    const auto dists = torchgis::en::dist(points, centroid);

    torch::Tensor values, indices;
    std::tie(values, indices) = at::topk(dists, 2, /*dim=*/-1, /*largest=*/false, /*sorted=1*/true);

    i0 = indices[0];
    i1 = indices[1];
  }

  // Find the third point such that forms the smallest circumcircle with i0 and i1.
  {
    const auto p0 = points[i0].unsqueeze(0);
    const auto p1 = points[i1].unsqueeze(0);

    const auto radiuses = torchgis::en::circumradius2d(points, p0, p1);

    torch::Tensor values, indices;
    std::tie(values, indices) = at::topk(radiuses, 1, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);
    i2 = indices[0];
  }

  //at::native::binary_kernel_reduce();


  return at::cat({points[i0], points[i1], points[i2]});
}


} // en
} // torchgis
