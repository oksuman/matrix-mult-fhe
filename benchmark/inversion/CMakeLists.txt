set(INVERSION_BENCHMARK_TARGETS
    benchmark_inversion_jkls18
    benchmark_inversion_rt22
    benchmark_inversion_as24
    benchmark_inversion_newcol
    benchmark_inversion_newrow
    benchmark_inversion_diag
    benchmark_inversion_naive
)

foreach(target ${INVERSION_BENCHMARK_TARGETS})
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target}
        PRIVATE
        matrix_operations
        utils
        benchmark::benchmark
        OPENFHEcore
        OPENFHEpke
    )
endforeach()

# Copy script files to build directory
configure_file(run_inversion_benchmarks.sh ${CMAKE_CURRENT_BINARY_DIR}/run_inversion_benchmarks.sh COPYONLY)