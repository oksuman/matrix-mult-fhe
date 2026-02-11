# Plaintext LDA executable (no encryption dependencies)
add_executable(lda_plaintext
    src/main_plaintext.cpp
)

target_include_directories(lda_plaintext
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(lda_plaintext
    PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

target_compile_features(lda_plaintext PRIVATE cxx_std_17)

# Encrypted LDA executable (AR24 and NewCol comparison)
add_executable(lda_encrypted
    src/main_encrypted.cpp
)

target_include_directories(lda_encrypted
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/third-party/include
)

target_link_libraries(lda_encrypted
    PRIVATE
    matrix_operations
    OPENFHEcore
    OPENFHEpke
    utils
)

target_compile_definitions(lda_encrypted
    PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

target_compile_features(lda_encrypted PRIVATE cxx_std_17)

# Quick inversion test (using pre-computed S_W)
add_executable(test_inversion_only
    src/test_inversion_only.cpp
)

target_include_directories(test_inversion_only
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/third-party/include
)

target_link_libraries(test_inversion_only
    PRIVATE
    matrix_operations
    OPENFHEcore
    OPENFHEpke
    utils
)

target_compile_definitions(test_inversion_only
    PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

target_compile_features(test_inversion_only PRIVATE cxx_std_17)

# Benchmark: NewCol vs AR24 comparison
add_executable(benchmark_lda
    src/benchmark_lda.cpp
)

target_include_directories(benchmark_lda
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/third-party/include
)

target_link_libraries(benchmark_lda
    PRIVATE
    matrix_operations
    OPENFHEcore
    OPENFHEpke
    utils
)

target_compile_definitions(benchmark_lda
    PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

target_compile_features(benchmark_lda PRIVATE cxx_std_17)
