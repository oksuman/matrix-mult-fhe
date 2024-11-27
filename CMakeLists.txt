cmake_minimum_required(VERSION 3.5.1)
project(matrix-mult-fhe CXX)

if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    endif()
endif()

# OpenMP package finding
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Compiler warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=c++17-attribute-extensions")
endif()

# Build options
option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_WERROR OFF)

# External dependencies
add_subdirectory(external/googletest)
add_subdirectory(external/benchmark)
enable_testing()

# OpenFHE
find_package(OpenFHE CONFIG REQUIRED)
if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif()

# OpenFHE compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenFHE_CXX_FLAGS}")

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
)

# Library directories
link_directories(
    ${OpenFHE_LIBDIR}
    ${OPENMP_LIBRARIES}
)

# Utils library (header-only)
add_library(utils INTERFACE)
target_include_directories(utils
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Matrix operations library
add_library(matrix_operations STATIC
    src/encryption.cpp
)

# Link dependencies
target_link_libraries(matrix_operations
    PUBLIC
    OPENFHEcore
    OPENFHEpke
    utils
    OpenMP::OpenMP_CXX  # Add OpenMP linking
)

# Set include directories for matrix_operations
target_include_directories(matrix_operations
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add project subdirectories
add_subdirectory(tests)
# add_subdirectory(benchmark)
add_subdirectory(utils)