# 실행 파일 생성
add_executable(convergence_test
    convergence_test.cpp
)

# 필요한 라이브러리 링크
target_link_libraries(convergence_test
    PRIVATE
    matrix_operations
)

# 실행 파일 위치 지정
set_target_properties(convergence_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 컴파일러 옵션 설정
target_compile_features(convergence_test PRIVATE cxx_std_17)