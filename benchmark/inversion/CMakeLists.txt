set(INVERSION_BENCHMARK_TARGETS
    benchmark_inversion_jkls18
    benchmark_inversion_rt22
    benchmark_inversion_ar24
    benchmark_inversion_newcol
    benchmark_inversion_naive
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

# Simplified inversion benchmarks (upperbound scaling, no trace/scalar-inverse)
set(INVERSION_SIMPLE_BENCHMARK_TARGETS
    benchmark_inversion_simple_jkls18
    benchmark_inversion_simple_rt22
    benchmark_inversion_simple_ar24
    benchmark_inversion_simple_newcol
)

foreach(target ${INVERSION_SIMPLE_BENCHMARK_TARGETS})
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
