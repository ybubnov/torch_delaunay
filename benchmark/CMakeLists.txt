find_package(benchmark REQUIRED)

# Define the c++ standard.
add_definitions(-std=c++17)
add_custom_target(benchmark)

include_directories()

file(GLOB torch_delaunay_BENCHMARK_SOURCES *.cc)

foreach(torch_delaunay_BENCHMARK_SOURCE ${torch_delaunay_BENCHMARK_SOURCES})
    # Extract extension from the file name.
    get_filename_component(torch_delaunay_BENCHMARK ${torch_delaunay_BENCHMARK_SOURCE} NAME_WE)

    add_executable(${torch_delaunay_BENCHMARK} ${torch_delaunay_BENCHMARK_SOURCE})
    target_link_libraries(${torch_delaunay_BENCHMARK} "${TORCH_LIBRARIES}")
    target_link_libraries(${torch_delaunay_BENCHMARK} torch_delaunay)
    target_link_libraries(${torch_delaunay_BENCHMARK} benchmark::benchmark)

    # Define a new test from the source file.
    add_dependencies(benchmark ${torch_delaunay_BENCHMARK})
    add_custom_command(TARGET benchmark POST_BUILD COMMAND ${torch_delaunay_BENCHMARK})
endforeach(torch_delaunay_BENCHMARK_SOURCE)
