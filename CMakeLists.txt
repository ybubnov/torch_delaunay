cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(torchgis)

include_directories(include)
include_directories(include/torchgis)

aux_source_directory(src torchgis_SOURCES)

add_library(torchgis SHARED ${torchgis_SOURCES})

find_package(Torch REQUIRED)

target_link_libraries(torchgis "${TORCH_LIBRARIES}")

set_property(TARGET torchgis PROPERTY CXX_STANDARD 14)
