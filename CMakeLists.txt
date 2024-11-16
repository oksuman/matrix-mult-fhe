cmake_minimum_required(VERSION 3.5.1)
project(matrix-mult-fhe CXX)
set(CMAKE_CXX_STANDARD 17)
option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)

# External dependencies from submodules
add_subdirectory(external/googletest)
add_subdirectory(external/benchmark)
set(BENCHMARK_ENABLE_TESTING OFF)  
enable_testing()

# OpenFHE
find_package(OpenFHE CONFIG REQUIRED)
if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif()

set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

# Include directories
include_directories(
    ${OPENMP_INCLUDES}
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/utils
    ${PROJECT_SOURCE_DIR}/app/linear_regression/include 
)

# Library directories
link_directories(
    ${OpenFHE_LIBDIR}
    ${OPENMP_LIBRARIES}
)

# Add utils library (header-only)
add_library(utils INTERFACE)
target_include_directories(utils 
    INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Source files
add_library(matrix_operations STATIC
    src/encryption.cpp
)

# Link against OpenFHE libraries
target_link_libraries(matrix_operations
    PUBLIC
    OPENFHEcore
    OPENFHEpke
    utils
)

# Set include directories for the library
target_include_directories(matrix_operations
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add subdirectories
add_subdirectory(tests)
add_subdirectory(benchmark)
add_subdirectory(app/linear_regression)