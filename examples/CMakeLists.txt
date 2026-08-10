# ─── Examples ─────────────────────────────────────────────────────────────────

# MNIST Inference: end-to-end digit classification
add_executable(mnist_inference mnist_inference.cpp)
target_include_directories(mnist_inference PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_compile_options(mnist_inference PRIVATE -O2)

# Generate Test Weights: creates synthetic model files (no PyTorch needed)
add_executable(generate_test_weights generate_test_weights.cpp)
target_include_directories(generate_test_weights PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
