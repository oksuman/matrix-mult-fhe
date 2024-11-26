find_package(OpenFHE CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

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
    naive_inversion.h
)

add_library(matrix_operations STATIC ${SOURCES} ${HEADERS})

target_link_libraries(matrix_operations
    PUBLIC
    OPENFHEcore
    OPENFHEpke
    OpenMP::OpenMP_CXX
)

target_include_directories(matrix_operations
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 컴파일러 최적화 플래그 추가
target_compile_options(matrix_operations
    PRIVATE
    -O3
    -march=native
)

if(OpenMP_CXX_FOUND)
    target_compile_options(matrix_operations
        PRIVATE
        ${OpenMP_CXX_FLAGS}
    )
endif()