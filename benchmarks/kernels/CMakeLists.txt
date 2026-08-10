include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# ─── MatMul Kernel Benchmark ─────────────────────────────────────────────────
# Compares Naive vs Tiled vs NEON across multiple matrix sizes.
# Reports GFLOPS for each configuration.
add_executable(bench_matmul bench_matmul.cpp)
target_include_directories(bench_matmul PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(bench_matmul PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
)

# Enable NEON on ARM (Apple Silicon)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_compile_options(bench_matmul PRIVATE -O3 -mcpu=native)
else()
    target_compile_options(bench_matmul PRIVATE -O3 -march=native)
endif()
