find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(Boost COMPONENTS json REQUIRED)

# Define the c++ standard.
add_definitions(-std=c++17)

include_directories()


file(GLOB torch_delaunay_TEST_SOURCES *.cc)

foreach(torch_delaunay_TEST_SOURCE ${torch_delaunay_TEST_SOURCES})
    # Extract extension from the file name.
    get_filename_component(torch_delaunay_TEST ${torch_delaunay_TEST_SOURCE} NAME_WE)

    add_executable(${torch_delaunay_TEST} ${torch_delaunay_TEST_SOURCE})
    target_link_libraries(${torch_delaunay_TEST} Boost::unit_test_framework)
    target_link_libraries(${torch_delaunay_TEST} Boost::json)
    target_link_libraries(${torch_delaunay_TEST} "${TORCH_LIBRARIES}")
    target_link_libraries(${torch_delaunay_TEST} torch_delaunay)

    target_compile_definitions(
        ${torch_delaunay_TEST}
        PUBLIC __lib_torch_delaunay_test_fixture_directory="${PROJECT_SOURCE_DIR}/test_fixture"
    )

    # Define a new test from the source file.
    add_test(NAME ${torch_delaunay_TEST} COMMAND ${torch_delaunay_TEST})
endforeach(torch_delaunay_TEST_SOURCE)
