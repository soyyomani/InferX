include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# ─── Memory Module Tests ──────────────────────────────────────────────────────
add_executable(test_memory test_memory.cpp)
target_include_directories(test_memory PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test_memory PRIVATE
    GTest::gtest_main
)
add_test(NAME test_memory COMMAND test_memory)
