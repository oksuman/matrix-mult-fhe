add_executable(convergence_test
    convergence_test.cpp
)

target_link_libraries(convergence_test
    PRIVATE
    matrix_operations
)

set_target_properties(convergence_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_features(convergence_test PRIVATE cxx_std_17)