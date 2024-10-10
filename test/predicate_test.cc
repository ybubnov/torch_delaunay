/// Copyright (C) 2024, Yakau Bubnou
///
/// This program is free software: you can redistribute it and/or modify
/// it under the terms of the GNU General Public License as published by
/// the Free Software Foundation, either version 3 of the License, or
/// (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

    auto triangle = torch::tensor({0, 1, 2});
    auto is_incircle = incircle2d(points.index({triangle}), points[3]);

    BOOST_REQUIRE(is_incircle);
}


BOOST_AUTO_TEST_CASE(test_incircle2d_batch)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

    auto p0 = torch::tensor({{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}}, options);
    auto p1 = torch::tensor({{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}}, options);
    auto p2 = torch::tensor({{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}, {0.0, 4.0}}, options);

    auto q = torch::tensor({{4.5, 4.9}, {2.5, 2.6}, {3.1, 3.2}, {8.2, 9.4}}, options);
    auto res = incircle2d(p0, p1, p2, q);

    BOOST_REQUIRE_EQUAL(res.sizes(), c10::IntArrayRef({4}));
}


BOOST_AUTO_TEST_SUITE_END()
