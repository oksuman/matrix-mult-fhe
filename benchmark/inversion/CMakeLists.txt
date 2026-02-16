set(INVERSION_BENCHMARK_TARGETS
    benchmark_inversion_jkls18
    benchmark_inversion_rt22
    benchmark_inversion_ar24
    benchmark_inversion_newcol
    benchmark_inversion_newrow
    benchmark_inversion_diag
    benchmark_inversion_naive
    debug_inversion
    debug_iteration
    test_newcol_all
    test_ar24_debug
    test_d32_all
    test_transpose_d32
    test_rt22_trace
)

foreach(target ${INVERSION_BENCHMARK_TARGETS})
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target}
        PRIVATE
        matrix_operations
        utils
        OPENFHEcore
        OPENFHEpke
    )
endforeach()

# Copy script files to build directory
configure_file(run_inversion_benchmarks.sh ${CMAKE_CURRENT_BINARY_DIR}/run_inversion_benchmarks.sh COPYONLY)
configure_file(generate_summary_table.sh ${CMAKE_CURRENT_BINARY_DIR}/generate_summary_table.sh COPYONLY)