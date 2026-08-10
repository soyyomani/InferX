include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# ─── Quantization Tests ───────────────────────────────────────────────────────
add_executable(test_quantize test_quantize.cpp)
target_include_directories(test_quantize PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test_quantize PRIVATE
    GTest::gtest_main
)
add_test(NAME test_quantize COMMAND test_quantize)
