#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadtree

#include <boost/test/included/unit_test.hpp>

#include <torch_delaunay.h>


using namespace torch_delaunay;


BOOST_AUTO_TEST_SUITE(TestShull)


BOOST_AUTO_TEST_CASE(test_shull2d)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto points = torch::tensor({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}, options);

    auto simplices = shull2d(points);
    std::cout << "SIMPLICES: " << simplices << std::endl;

    BOOST_CHECK_EQUAL(simplices.sizes(), torch::IntArrayRef({1, 3}));
}


BOOST_AUTO_TEST_SUITE_END()
