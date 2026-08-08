# WASM build for neural network visualization modules
# Build with: emcmake cmake .. && emmake make

cmake_minimum_required(VERSION 3.20)

add_executable(inferx_nn_wasm nn_bindings.cpp)

target_include_directories(inferx_nn_wasm PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(inferx_nn_wasm PUBLIC cxx_std_20)

if(EMSCRIPTEN)
    set_target_properties(inferx_nn_wasm PROPERTIES
        OUTPUT_NAME "inferx_nn_wasm"
        SUFFIX ".js"
    )

    target_link_options(inferx_nn_wasm PRIVATE
        --bind
        -s MODULARIZE=1
        -s EXPORT_NAME="createInferXNNModule"
        -s ALLOW_MEMORY_GROWTH=1
        -s ENVIRONMENT=web
        -s SINGLE_FILE=0
        -O2
    )
endif()
