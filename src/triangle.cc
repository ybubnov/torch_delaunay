// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <torch_delaunay/predicate.h>


using namespace torch::indexing;


namespace torch_delaunay {


std::tuple<torch::Tensor, torch::Tensor>
_cc_coordinates(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    const auto a = p0;
    const auto b = -p0 + p1;
    const auto c = -p0 + p2;

    const std::initializer_list<at::indexing::TensorIndex> x = {Slice(), 0};
    const std::initializer_list<at::indexing::TensorIndex> y = {Slice(), 1};
    const auto d = 2 * (b.index(x) * c.index(y) - b.index(y) * c.index(x));

    const auto b_norm = b.square().sum({1});
    const auto c_norm = c.square().sum({1});

    const auto ux = (c.index(y) * b_norm - b.index(y) * c_norm) / d;
    const auto uy = (b.index(x) * c_norm - c.index(x) * b_norm) / d;

    return std::forward_as_tuple(ux, uy);
}


torch::Tensor
circumcenter2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    const auto [ux, uy] = _cc_coordinates(p0, p1, p2);
    return at::column_stack({ux, uy}) + p0;
}


torch::Tensor
circumcenter2d(const torch::Tensor& points)
{
    return circumcenter2d(
        points.index({Slice(), 0, Slice()}), points.index({Slice(), 1, Slice()}),
        points.index({Slice(), 2, Slice()})
    );
}


torch::Tensor
circumradius2d(const torch::Tensor& p0, const torch::Tensor& p1, const torch::Tensor& p2)
{
    constexpr std::string_view op = "circumradius2d";

    TORCH_CHECK(p0.dim() == 2, op, " only supports 2D tensors, got: ", p0.dim(), "D");
    TORCH_CHECK(p1.dim() == 2, op, " only supports 2D tensors, got: ", p1.dim(), "D");
    TORCH_CHECK(p2.dim() == 2, op, " only supports 2D tensors, got: ", p2.dim(), "D");

    TORCH_CHECK(p0.size(1) == 2, op, " only supports 2D coordinates, got: ", p0.size(1), "D");
    TORCH_CHECK(p1.size(1) == 2, op, " only supports 2D coordinates, got: ", p0.size(1), "D");
    TORCH_CHECK(p2.size(1) == 2, op, " only supports 2D coordinates, got: ", p0.size(1), "D");

    const auto [ux, uy] = _cc_coordinates(p0, p1, p2);
    return (ux.square() + uy.square()).sqrt();
}


torch::Tensor
circumradius2d(const torch::Tensor& points)
{
    return circumradius2d(
        points.index({Slice(), 0, Slice()}), points.index({Slice(), 1, Slice()}),
        points.index({Slice(), 2, Slice()})
    );
}


} // namespace torch_delaunay
