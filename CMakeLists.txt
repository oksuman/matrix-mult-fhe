cmake_minimum_required(VERSION 3.5.1)
project(matrix-mult-fhe CXX)

# Basic project settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release FORCE)

# Performance optimization settings
set(WITH_NATIVEOPT ON FORCE)
set(NATIVE_SIZE 64 CACHE STRING "Word size for small integers (32/64)") # 64 for general use, 32 for moduli <= 28 bits
option(WITH_OPENMP "Enable OpenMP support" ON)

# Build options
option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)
option(ENABLE_PRINT_PT "Enable PRINT_PT macro" OFF)

# Find required packages
find_package(OpenFHE CONFIG REQUIRED)
if(OpenFHE_FOUND)
    message(STATUS "Found OpenFHE")
else()
    message(FATAL_ERROR "OpenFHE not found")
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${OpenFHE_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
endif()

# Include directories
include_directories(
    ${OPENMP_INCLUDES}
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Library directories
link_directories(
    ${OpenFHE_LIBDIR}
    ${OPENMP_LIBRARIES}
)

# Source files
set(SOURCES
    encryption.cpp
    main.cpp
)

# Header files
set(HEADERS
    encryption.h
    mat_inv.h
    matrix_algo_singlePack.h
    matrix_inversion_algo.h
)

# Main executable
add_executable(app ${SOURCES} ${HEADERS})

# Link libraries
if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    target_link_libraries(app PRIVATE ${OpenFHE_STATIC_LIBRARIES} OpenMP::OpenMP_CXX)
else()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    target_link_libraries(app PRIVATE ${OpenFHE_SHARED_LIBRARIES} OpenMP::OpenMP_CXX)
endif()

# Enable PRINT_PT macro if requested
if(ENABLE_PRINT_PT)
    target_compile_definitions(app PRIVATE ENABLE_PRINT_PT)
endif()

# Display important configuration information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Native optimizations: ${WITH_NATIVEOPT}")
message(STATUS "Native size: ${NATIVE_SIZE}")
message(STATUS "OpenMP enabled: ${WITH_OPENMP}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")