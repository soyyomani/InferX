add_library(inferx_tensor STATIC)

target_sources(inferx_tensor PRIVATE
    shape.cpp
    stride.cpp
    storage.cpp
    broadcast.cpp
)

target_include_directories(inferx_tensor PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(inferx_tensor PUBLIC cxx_std_20)

# Alias for cleaner usage
add_library(InferX::Tensor ALIAS inferx_tensor)
