// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE torch_delaunay

#include <boost/test/included/unit_test.hpp>

#include <torch_delaunay.h>


using namespace torch_delaunay;


BOOST_AUTO_TEST_SUITE(TestPredicate)


BOOST_AUTO_TEST_CASE(test_incircle2d)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto points = torch::tensor({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {4.0, 4.0}}, options);
    auto P = points.accessor<double, 2>();

    auto is_incircle = incircle2d_kernel(P[0], P[1], P[2], P[3]);
    BOOST_REQUIRE(is_incircle);
}


BOOST_AUTO_TEST_CASE(test_incircle2d_batch)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

    auto points0 = torch::tensor({{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}}, options);
    auto points1 = torch::tensor({{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}}, options);
    auto points2 = torch::tensor({{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}, {0.0, 4.0}}, options);
    auto queries = torch::tensor({{4.5, 4.9}, {2.5, 2.6}, {3.1, 3.2}, {8.2, 9.4}}, options);

    auto P0 = points0.accessor<double, 2>();
    auto P1 = points1.accessor<double, 2>();
    auto P2 = points2.accessor<double, 2>();
    auto Q = queries.accessor<double, 2>();

    for (std::size_t i = 0; i < 4; i++) {
        auto is_incircle = incircle2d_kernel(P0[i], P1[i], P2[i], Q[i]);
        BOOST_REQUIRE(is_incircle);
    }
}


BOOST_AUTO_TEST_SUITE_END()
