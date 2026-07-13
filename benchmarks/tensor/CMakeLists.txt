include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# Helper function to add a tensor benchmark
function(add_tensor_benchmark BENCH_NAME)
    add_executable(${BENCH_NAME} ${BENCH_NAME}.cpp)
    target_link_libraries(${BENCH_NAME} PRIVATE
        InferX::Tensor
        benchmark::benchmark
        benchmark::benchmark_main
    )
endfunction()

# Benchmarks will be added as components are implemented:
# add_tensor_benchmark(bench_allocation)
# add_tensor_benchmark(bench_access)
# add_tensor_benchmark(bench_iteration)
# add_tensor_benchmark(bench_operations)
