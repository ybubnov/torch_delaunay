/// SPDX-License-Identifier: GPL-3.0-or-later
/// SPDX-FileCopyrightText: 2025 Yakau Bubnou
/// SPDX-FileType: SOURCE

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE torch_delaunay

#include <boost/test/included/unit_test.hpp>

#include <torch_delaunay.h>

#include "testing.h"


using namespace torch_delaunay;


BOOST_AUTO_TEST_SUITE(TestTriangle)


BOOST_AUTO_TEST_CASE(test_circumradius2d_wrong_dim)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto p0 = torch::tensor({0.5, 0.5}, options);
    auto p1 = torch::tensor({0.7, 0.7}, options);
    auto points = torch::tensor({{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}}, options);

    BOOST_CHECK_EXCEPTION(
        circumradius2d(p0, p1, points), c10::Error,
        exception_contains_text<c10::Error>("only supports 2D tensors")
    );

    BOOST_CHECK_EXCEPTION(
        circumradius2d(p0.view({-1, 2}), p1.view({-1, 2}), points.view({2, 3})), c10::Error,
        exception_contains_text<c10::Error>("only supports 2D coordinates")
    );
}


BOOST_AUTO_TEST_CASE(test_circumradius2d)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto p0 = torch::tensor({0.5, 0.5}, options);
    auto p1 = torch::tensor({0.7, 0.7}, options);
    auto points = torch::tensor({{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}}, options);

    auto radii = circumradius2d(p0.view({-1, 2}), p1.view({-1, 2}), points);

    BOOST_CHECK_EQUAL(radii.sizes(), torch::IntArrayRef({3}));
}


BOOST_AUTO_TEST_SUITE_END()
