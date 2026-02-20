# Plaintext logistic regression (Fixed Hessian)
add_executable(fh_plaintext
    src/main_plaintext.cpp
)

target_include_directories(fh_plaintext
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(fh_plaintext
    PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/dataset"
)

target_compile_features(fh_plaintext PRIVATE cxx_std_17)

# Encrypted logistic regression (Fixed Hessian) — Heart Disease
add_executable(fh_encrypted
    src/main_encrypted.cpp
)

target_include_directories(fh_encrypted
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/third-party/include
)

target_link_libraries(fh_encrypted
    PRIVATE
    matrix_operations
    OPENFHEcore
    OPENFHEpke
)

target_compile_definitions(fh_encrypted
    PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/dataset"
    DATASET_HEART
)

target_compile_features(fh_encrypted PRIVATE cxx_std_17)

# Encrypted logistic regression (Fixed Hessian) — Diabetes
add_executable(fh_encrypted_diabetes
    src/main_encrypted.cpp
)

target_include_directories(fh_encrypted_diabetes
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/third-party/include
)

target_link_libraries(fh_encrypted_diabetes
    PRIVATE
    matrix_operations
    OPENFHEcore
    OPENFHEpke
)

target_compile_definitions(fh_encrypted_diabetes
    PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/dataset"
    DATASET_DIABETES
)

target_compile_features(fh_encrypted_diabetes PRIVATE cxx_std_17)

# Plaintext logistic regression — Diabetes
add_executable(fh_plaintext_diabetes
    src/main_plaintext.cpp
)

target_include_directories(fh_plaintext_diabetes
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(fh_plaintext_diabetes
    PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/dataset"
    DATASET_DIABETES
)

target_compile_features(fh_plaintext_diabetes PRIVATE cxx_std_17)

