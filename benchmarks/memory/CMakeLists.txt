include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# ─── Memory Allocation Benchmark ─────────────────────────────────────────────
# Compares Arena vs BufferPool vs system malloc for inference allocation patterns.
add_executable(bench_memory bench_memory.cpp)
target_include_directories(bench_memory PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(bench_memory PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
)
target_compile_options(bench_memory PRIVATE -O3)
