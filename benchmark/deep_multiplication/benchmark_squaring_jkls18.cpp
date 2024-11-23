#include "benchmark_squaring.h"

const int ITERATION_COUNT = 1;

BENCHMARK_TEMPLATE(BM_JKLS18_Squaring, 4)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_JKLS18_Squaring, 8)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_JKLS18_Squaring, 16)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_JKLS18_Squaring, 32)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
// BENCHMARK_TEMPLATE(BM_JKLS18_Squaring, 64)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);

BENCHMARK_MAIN();