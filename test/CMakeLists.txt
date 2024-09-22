find_package(Boost COMPONENTS unit_test_framework REQUIRED)

# Define the c++ standard.
add_definitions(-std=c++17)

include_directories()


file(GLOB torch_delaunay_TEST_SOURCES *.cc)

foreach(torch_delaunay_TEST_SOURCE ${torch_delaunay_TEST_SOURCES})
    # Extract extension from the file name.
    get_filename_component(torch_delaunay_TEST ${torch_delaunay_TEST_SOURCE} NAME_WE)

    add_executable(${torch_delaunay_TEST} ${torch_delaunay_TEST_SOURCE})
    target_link_libraries(${torch_delaunay_TEST} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    target_link_libraries(${torch_delaunay_TEST} "${TORCH_LIBRARIES}")
    target_link_libraries(${torch_delaunay_TEST} torch_delaunay)

    # Define a new test from the source file.
    add_test(NAME ${torch_delaunay_TEST} COMMAND ${torch_delaunay_TEST})
endforeach(torch_delaunay_TEST_SOURCE)
