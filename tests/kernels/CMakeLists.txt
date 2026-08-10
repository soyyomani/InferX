include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# ─── Kernel Tests ─────────────────────────────────────────────────────────────
add_executable(test_matmul_kernels test_matmul_kernels.cpp)
target_include_directories(test_matmul_kernels PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test_matmul_kernels PRIVATE
    GTest::gtest_main
)
add_test(NAME test_matmul_kernels COMMAND test_matmul_kernels)

# Enable optimizations for NEON testing (must match production flags)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_compile_options(test_matmul_kernels PRIVATE -O2 -mcpu=native)
else()
    target_compile_options(test_matmul_kernels PRIVATE -O2 -march=native)
endif()
