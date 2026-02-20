set(SQUARING_BENCHMARK_TARGETS
    benchmark_squaring_jkls18
    benchmark_squaring_rt22
    benchmark_squaring_ar24
    benchmark_squaring_newcol
    benchmark_squaring_naive
)

# Memory tracker source
set(MEMORY_TRACKER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../memory_tracker.cpp)

foreach(target ${SQUARING_BENCHMARK_TARGETS})
    add_executable(${target} ${target}.cpp ${MEMORY_TRACKER_SRC})
    target_link_libraries(${target}
        PRIVATE
        matrix_operations
        utils
        OPENFHEcore
        OPENFHEpke
    )
endforeach()

configure_file(run_squaring_benchmarks.sh ${CMAKE_CURRENT_BINARY_DIR}/run_squaring_benchmarks.sh COPYONLY)
configure_file(generate_summary_table.sh ${CMAKE_CURRENT_BINARY_DIR}/generate_summary_table.sh COPYONLY)
