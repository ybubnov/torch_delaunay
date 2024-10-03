#include <stack>
#include <unordered_map>

#include <torch_delaunay/predicate.h>
#include <torch_delaunay/sweephull.h>
#include <torch_delaunay/triangle.h>

#include <ATen/native/cpu/Loops.h>
#include <torch/linalg.h>


using namespace torch::indexing;


namespace torch_delaunay {


std::tuple<torch::Tensor, torch::Tensor>
_cc_coordinates(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
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

    return std::forward_as_tuple(ux, uy);
}


torch::Tensor
circumcenter2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    auto [ux, uy] = _cc_coordinates(p0, p1, p2);
    return at::column_stack({ux, uy}) + p0;
}


// TODO: change circumradius implementation, so it does not produce nans and returns 0 instead.
torch::Tensor
circumradius2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    const std::string op = "circumcenter2d";
    TORCH_CHECK(p0.dim() == 2, op, " only supports 2D tensors, got: ", p0.dim(), "D");
    TORCH_CHECK(p1.dim() == 2, op, " only supports 2D tensors, got: ", p1.dim(), "D");
    TORCH_CHECK(p2.dim() == 2, op, " only supports 2D tensors, got: ", p2.dim(), "D");

    TORCH_CHECK(p0.size(1) == 2, op, " only supports 2D coordinates, got: ", p0.size(1), "D");
    TORCH_CHECK(p1.size(1) == 2, op, " only supports 2D coordinates, got: ", p0.size(1), "D");
    TORCH_CHECK(p2.size(1) == 2, op, " only supports 2D coordinates, got: ", p0.size(1), "D");

    auto [ux, uy] = _cc_coordinates(p0, p1, p2);
    return (ux.square() + uy.square()).sqrt();
}


} // namespace torch_delaunay
