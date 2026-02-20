# Plaintext LR (no encryption)
add_executable(lr_plaintext
    src/main_plaintext.cpp
)

target_include_directories(lr_plaintext
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/utils
)

target_compile_definitions(lr_plaintext
    PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/dataset"
)

target_compile_features(lr_plaintext PRIVATE cxx_std_17)

# Main LR benchmark (Naive + AR24 + NewCol comparison)
add_executable(lr_benchmark
    src/main_naive_vs_packed.cpp
    ${CMAKE_SOURCE_DIR}/benchmark/memory_tracker.cpp
)

target_include_directories(lr_benchmark
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/app/common
    ${CMAKE_SOURCE_DIR}/benchmark
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/third-party/include
)

target_link_libraries(lr_benchmark
    PRIVATE
    matrix_operations
    OPENFHEcore
    OPENFHEpke
    utils
)

target_compile_definitions(lr_benchmark
    PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/dataset"
)

target_compile_features(lr_benchmark PRIVATE cxx_std_17)
