cmake_minimum_required(VERSION 3.20)
project(InferX
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "High-performance C++20 AI inference engine"
)

# ─── Global Settings ─────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings
add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -Werror=return-type
)

# ─── Options ─────────────────────────────────────────────────────────────────
option(INFERX_BUILD_TESTS "Build unit tests" ON)
option(INFERX_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(INFERX_BUILD_EXAMPLES "Build examples" ON)

# ─── Library ─────────────────────────────────────────────────────────────────
add_subdirectory(src/tensor)

# ─── Tests ───────────────────────────────────────────────────────────────────
if(INFERX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/tensor)
    add_subdirectory(tests/kernels)
    add_subdirectory(tests/memory)
    add_subdirectory(tests/graph)
    add_subdirectory(tests/quantize)
    add_subdirectory(tests/parallel)
endif()

# ─── Benchmarks ──────────────────────────────────────────────────────────────
if(INFERX_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks/tensor)
    add_subdirectory(benchmarks/kernels)
    add_subdirectory(benchmarks/memory)
endif()

# ─── Examples ─────────────────────────────────────────────────────────────────
if(INFERX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
