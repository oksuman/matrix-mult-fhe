# Tests directory CMake configuration
enable_testing()

# Find OpenFHE package (redundant but safe)
find_package(OpenFHE CONFIG REQUIRED)

# Define test sources
set(TEST_SOURCES
    matrix_operation_test.cpp

    mult_jkls18_test.cpp
    mult_rt22_test.cpp  
    mult_as24_test.cpp
    mult_newRow_test.cpp
    mult_newCol_test.cpp

    inverse_diagonal_test.cpp
    inverse_jkls18_test.cpp
    inverse_rt22_test.cpp
    inverse_as24_test.cpp
    inverse_newCol_test.cpp
    inverse_newRow_test.cpp
)

# Add each test executable
foreach(test_source ${TEST_SOURCES})
    # Get the filename without extension
    get_filename_component(test_name ${test_source} NAME_WE)
    
    # Create executable
    add_executable(${test_name} ${test_source})
    
    # Link against necessary libraries
    target_link_libraries(${test_name}
        PRIVATE
        matrix_operations
        utils
        OPENFHEcore
        OPENFHEpke
        GTest::gtest
        GTest::gtest_main
    )
    
    # Add the test to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()