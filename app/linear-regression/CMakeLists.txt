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
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

target_compile_features(lr_plaintext PRIVATE cxx_std_17)

# Encrypted LR - NewCol
add_executable(lr_newcol
    src/main_newcol.cpp
)

add_executable(lr_ar24
     src/main_ar24.cpp
)

target_include_directories(lr_newcol
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/third-party/include
)

target_link_libraries(lr_newcol
    PRIVATE
    matrix_operations
    OPENFHEcore
    OPENFHEpke
    utils
)

target_compile_definitions(lr_newcol
    PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

target_compile_features(lr_newcol PRIVATE cxx_std_17)

target_include_directories(lr_ar24
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/third-party/include
)

target_link_libraries(lr_ar24
    PRIVATE
    matrix_operations
    OPENFHEcore
    OPENFHEpke
    utils
)

target_compile_definitions(lr_ar24
    PRIVATE
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

target_compile_features(lr_ar24 PRIVATE cxx_std_17)