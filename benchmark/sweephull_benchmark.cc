// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

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
