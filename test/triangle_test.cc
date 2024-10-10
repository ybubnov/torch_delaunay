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
