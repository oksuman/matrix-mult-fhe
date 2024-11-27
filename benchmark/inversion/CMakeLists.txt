set(INVERSION_BENCHMARK_TARGETS
    benchmark_inversion_as24
    benchmark_inversion_newcol
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

