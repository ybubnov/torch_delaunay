// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <filesystem>
#include <fstream>
#include <functional>

#include <boost/json.hpp>
#include <torch/torch.h>


template <typename Exception>
std::function<bool(const Exception&)>
exception_contains_text(const std::string error_message)
{
    return [&](const Exception& error) -> bool {
        return std::string(error.what()).find(error_message) != std::string::npos;
    };
}


#ifndef __lib_torch_delaunay_test_fixture_directory
#define __lib_torch_delaunay_test_fixture_directory "test_fixture"
#endif


std::filesystem::path
test_fixture_path()
{
    return std::filesystem::path(__lib_torch_delaunay_test_fixture_directory);
}


torch::Tensor
make_fixture_points()
{
    boost::system::error_code json_error;
    boost::json::parse_options json_options;

    std::ifstream points_file(test_fixture_path() / "points.json");

    json_options.numbers = boost::json::number_precision::precise;
    auto json_value = boost::json::parse(points_file, json_error, {}, json_options);
    if (json_error.failed()) {
        throw std::runtime_error("failed parsing 'points.json' " + json_error.what());
    }

    const auto& points_array = json_value.as_array();
    auto points_options = torch::TensorOptions().dtype(torch::kFloat64);
    auto points_size = static_cast<long>(points_array.size());
    auto points = torch::empty({points_size, 2}, points_options);

    std::size_t i = 0;
    for (const auto& point : points_array) {
        std::size_t j = 0;
        for (const auto& coordinate : point.as_array()) {
            points[i][j++] = coordinate.as_double();
        }
        i++;
    }

    return points;
}
