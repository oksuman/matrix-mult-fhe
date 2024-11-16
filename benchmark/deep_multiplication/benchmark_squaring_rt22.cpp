#include "benchmark_squaring.h"

const int ITERATION_COUNT = 1;

BENCHMARK_TEMPLATE(BM_RT22_Squaring, 4)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_RT22_Squaring, 8)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_RT22_Squaring, 16)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_RT22_Squaring, 32)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);  // This will do both 32x32 and 64x64

BENCHMARK_MAIN();