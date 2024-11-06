# Source directory CMake configuration
find_package(OpenFHE CONFIG REQUIRED)

set(SOURCES
    encryption.cpp
)

set(HEADERS
    encryption.h
    mat_inv.h
    matrix_algo_multiPack.h
    matrix_algo_singlePack.h
    matrix_inversion_algo.h
    rotation.h
)

add_library(matrix_operations STATIC ${SOURCES} ${HEADERS})

target_link_libraries(matrix_operations
    PUBLIC
    OPENFHEcore
    OPENFHEpke
)

target_include_directories(matrix_operations
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)