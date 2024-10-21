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

#include <benchmark/benchmark.h>

#include <torch_delaunay.h>


using namespace torch_delaunay;


static void
benchmark_shull2d_uniform_float64(benchmark::State& state)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto points = torch::rand({state.range(0), 2}, options);

    for (auto _ : state) {
        shull2d(points);
    }
}


static void
benchmark_shull2d_uniform_float32(benchmark::State& state)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto points = torch::rand({state.range(0), 2}, options);

    for (auto _ : state) {
        shull2d(points);
    }
}


static void
benchmark_shull2d_normal_float64(benchmark::State& state)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto points = torch::randn({state.range(0), 2}, options);

    for (auto _ : state) {
        shull2d(points);
    }
}


static void
benchmark_shull2d_normal_float32(benchmark::State& state)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto points = torch::randn({state.range(0), 2}, options);

    for (auto _ : state) {
        shull2d(points);
    }
}


BENCHMARK(benchmark_shull2d_uniform_float64)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Unit(benchmark::kMillisecond);


BENCHMARK(benchmark_shull2d_uniform_float32)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Unit(benchmark::kMillisecond);


BENCHMARK(benchmark_shull2d_normal_float64)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Unit(benchmark::kMillisecond);


BENCHMARK(benchmark_shull2d_normal_float32)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();
