cmake_minimum_required(VERSION 3.5.1)
project(matrix-mult-fhe CXX)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)

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
   ${PROJECT_SOURCE_DIR}/app/linear_regression/include 
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
)

# Set include directories for matrix_operations
target_include_directories(matrix_operations
   PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add project subdirectories
add_subdirectory(tests)
add_subdirectory(benchmark)
add_subdirectory(app/linear-regression)
add_subdirectory(app/linear-discriminant-analysis)
add_subdirectory(app/fixed-hessian)
add_subdirectory(utils)