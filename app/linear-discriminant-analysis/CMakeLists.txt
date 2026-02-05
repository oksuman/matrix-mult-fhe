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
