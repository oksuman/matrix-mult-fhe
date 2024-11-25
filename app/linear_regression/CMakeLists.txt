add_executable(lr_newcol
    src/main_newcol.cpp
)

add_executable(lr_as24
     src/main_as24.cpp
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

target_include_directories(lr_as24
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/third-party/include
)

target_link_libraries(lr_as24
    PRIVATE
    matrix_operations
    OPENFHEcore
    OPENFHEpke
    utils
)