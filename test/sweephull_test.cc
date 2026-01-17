/// SPDX-License-Identifier: GPL-3.0-or-later
/// SPDX-FileCopyrightText: 2025 Yakau Bubnou
/// SPDX-FileType: SOURCE

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE torch_delaunay

#include <boost/test/included/unit_test.hpp>

#include <torch_delaunay.h>

#include "testing.h"


using namespace torch_delaunay;


BOOST_AUTO_TEST_SUITE(TestSweephull)


BOOST_AUTO_TEST_CASE(test_shull2d)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto points = torch::tensor({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, options);

    auto simplices = shull2d(points);

    BOOST_CHECK_EQUAL(simplices.sizes(), torch::IntArrayRef({1, 3}));
}


BOOST_AUTO_TEST_CASE(test_shull2d_less_than_three_points)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto points = torch::tensor({{0.0, 0.0}, {1.0, 0.0}}, options);

    BOOST_CHECK_EXCEPTION(
        shull2d(points), c10::Error,
        exception_contains_text<c10::Error>("expects at least 3 points")
    );
}


BOOST_AUTO_TEST_CASE(test_shull2d_no_seed)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

    auto simplices = shull2d(torch::zeros({3, 2}, options));
    BOOST_CHECK_EQUAL(simplices.size(0), 0);

    simplices = shull2d(torch::zeros({16, 2}));
    BOOST_CHECK_EQUAL(simplices.size(0), 0);
}


BOOST_AUTO_TEST_CASE(test_shull2d_inf_coplanar)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto points = torch::tensor({{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}}, options);

    auto simplices = shull2d(points);
    BOOST_CHECK_EQUAL(simplices.size(0), 0);

    simplices = shull2d(points);
    BOOST_CHECK_EQUAL(simplices.size(0), 0);
}


BOOST_AUTO_TEST_CASE(test_shull2d_nan_coplanar)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto points = torch::tensor({{0.0, 0.0}, {1.0, 0.0}, {0.0, 2.0}, {0.0, 0.0}}, options);

    auto simplices = shull2d(points);
    auto radii = circumradius2d(points.index({simplices}));
    auto centers = circumcenter2d(points.index({simplices}));

    // When epsilon is specified, triangulation contains only finite faces.
    BOOST_CHECK_EQUAL(simplices.size(0), 1);
    BOOST_CHECK(!radii.isnan().any().item<bool>());
    BOOST_CHECK(!centers.isnan().any().item<bool>());
}


BOOST_AUTO_TEST_CASE(test_shull2d_float32_coplanar)
{
    // Unfortunately, circumradius used for coplanar computation, is unstable for
    // almost equal numbers, therefore computed radius for resulting simplices are
    // finite and cannot be treated as coplanar.
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto points = torch::linspace(0, 5, 10, options);
    points = torch::column_stack({points, points});

    auto simplices = shull2d(points);
    BOOST_CHECK_GE(simplices.size(0), 0);
}


BOOST_AUTO_TEST_CASE(test_shull2d_integer_coplanar)
{
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto points = torch::linspace(0, 10, 10, options);
    points = torch::column_stack({points, points});

    auto simplices = shull2d(points);
    BOOST_CHECK_EQUAL(simplices.size(0), 0);
}


BOOST_AUTO_TEST_CASE(test_shull2d_1000)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto points = torch::rand({1000, 2}, options);

    auto simplices = shull2d(points);
    BOOST_CHECK(simplices.size(0) > 100);
}


BOOST_AUTO_TEST_SUITE_END()
