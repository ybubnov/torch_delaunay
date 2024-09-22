cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(torch_delaunay, CXX)
add_definitions(-std=c++17)

find_package(Torch CONFIG REQUIRED)

enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(torch_delaunay_LIBRARY_VERSION "1.0.0")
set(torch_delaunay_LIBRARY_SOVERSION "1")


include_directories(include)
include_directories(include/torch_delaunay)
include_directories(src)


aux_source_directory(src torch_delaunay_SOURCES)

# Add the testing directory.
add_subdirectory(test)


add_library(torch_delaunay SHARED ${torch_delaunay_SOURCES})
set_target_properties(torch_delaunay PROPERTIES CXX_STANDARD 17)
set_target_properties(torch_delaunay PROPERTIES VERSION ${torch_delaunay_LIBRARY_VERSION})
set_target_properties(torch_delaunay PROPERTIES SOVERSION ${torch_delaunay_LIBRARY_SOVERSION})
target_link_libraries(torch_delaunay "${TORCH_LIBRARIES}")


# Install the library files.
install(DIRECTORY include/torch_delaunay DESTINATION torch_delaunay)
install(FILES include/torch_delaunay.h DESTINATION include)
install(TARGETS torch_delaunay DESTINATION lib)
