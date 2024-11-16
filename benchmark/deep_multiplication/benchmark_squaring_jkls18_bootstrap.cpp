#include "benchmark_squaring.h"

const int ITERATION_COUNT = 1;

BENCHMARK_TEMPLATE(BM_JKLS18_Squaring_Bootstrap, 4)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_JKLS18_Squaring_Bootstrap, 8)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_JKLS18_Squaring_Bootstrap, 16)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);
BENCHMARK_TEMPLATE(BM_JKLS18_Squaring_Bootstrap, 32)->Unit(benchmark::kSecond)->UseRealTime()->Iterations(ITERATION_COUNT);

BENCHMARK_MAIN();