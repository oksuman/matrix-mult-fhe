set(SQUARING_BENCHMARK_TARGETS
    benchmark_squaring_jkls18
    benchmark_squaring_rt22
    benchmark_squaring_ar24
    benchmark_squaring_newcol
    benchmark_squaring_newrow
    benchmark_squaring_diag
)

foreach(target ${SQUARING_BENCHMARK_TARGETS})
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target}
        PRIVATE
        matrix_operations
        utils
        benchmark::benchmark
        benchmark::benchmark_main  
        OPENFHEcore
        OPENFHEpke
    )
endforeach()

configure_file(run_squaring_benchmarks.sh ${CMAKE_CURRENT_BINARY_DIR}/run_squaring_benchmarks.sh COPYONLY)