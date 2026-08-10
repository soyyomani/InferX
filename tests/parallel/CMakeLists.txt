include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# ─── Parallel Module Tests ────────────────────────────────────────────────────
add_executable(test_parallel test_thread_pool.cpp)
target_include_directories(test_parallel PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test_parallel PRIVATE
    GTest::gtest_main
)
add_test(NAME test_parallel COMMAND test_parallel)
