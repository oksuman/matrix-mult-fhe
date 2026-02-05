set(BENCHMARK_TARGETS
    benchmark_jkls18
    benchmark_rt22
    benchmark_ar24
    benchmark_newcol
    benchmark_newrow
    benchmark_diag
)

foreach(target ${BENCHMARK_TARGETS})
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
configure_file(run_benchmarks.sh ${CMAKE_CURRENT_BINARY_DIR}/run_benchmarks.sh COPYONLY)