cmake_minimum_required(VERSION 3.20)
project(InferXWasm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(inferx_wasm
    bindings.cpp
    ../src/tensor/shape.cpp
    ../src/tensor/stride.cpp
    ../src/tensor/storage.cpp
    ../src/tensor/broadcast.cpp
)

target_include_directories(inferx_wasm PRIVATE
    ../include
)

set_target_properties(inferx_wasm PROPERTIES
    SUFFIX ".js"
    LINK_FLAGS "--bind -s MODULARIZE=1 -s EXPORT_NAME=createInferXModule -s ENVIRONMENT=web -s ALLOW_MEMORY_GROWTH=1 -s NO_EXIT_RUNTIME=1 -O2"
)
