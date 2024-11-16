add_executable(lr_newcol
    src/main.cpp
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

# Make sure data directory exists
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data)

# Copy data files if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/trainSet.csv)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/trainSet.csv
         DESTINATION ${CMAKE_BINARY_DIR}/data)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/testSet.csv)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/testSet.csv
         DESTINATION ${CMAKE_BINARY_DIR}/data)
endif()