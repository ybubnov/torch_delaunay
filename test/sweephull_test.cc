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

    BOOST_CHECK_EXCEPTION(
        shull2d(torch::zeros({3, 2}, options)), c10::Error,
        exception_contains_text<c10::Error>("missing third point for an initial simplex")
    );

    BOOST_CHECK_EXCEPTION(
        shull2d(torch::zeros({16, 2})), c10::Error,
        exception_contains_text<c10::Error>("missing third point for an initial simplex")
    );
}


BOOST_AUTO_TEST_CASE(test_shull2d_flat_simplex)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto points = torch::tensor({{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}}, options);

    BOOST_CHECK_EXCEPTION(
        shull2d(points), c10::Error, exception_contains_text<c10::Error>("encountered flat simplex")
    );
}


BOOST_AUTO_TEST_SUITE_END()
