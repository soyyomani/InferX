/// @file mnist_inference.cpp
/// @brief End-to-end MNIST digit classification using the InferX engine.
///
/// This demonstrates the complete inference pipeline:
///   1. Load trained model weights (from Python export)
///   2. Build computational graph (Input → MatMul+Bias → ReLU → MatMul+Bias → Softmax)
///   3. Load test images
///   4. Run inference through the graph executor (arena-backed memory)
///   5. Report accuracy
///
/// Architecture: 784 → 128 (ReLU) → 10 (Softmax) → predicted digit
///
/// Usage:
///   ./mnist_inference <model_dir>
///   ./mnist_inference models/
///
/// Expected output with trained weights:
///   Accuracy: 97%+ on 100 test images

#include <inferx/model/model_loader.h>
#include <inferx/memory/arena.h>

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>

using namespace inferx;

/// Perform matrix-vector multiply: y = W × x + b
/// W is [output_dim × input_dim], x is [input_dim], y is [output_dim]
static void linear_forward(const float* W, const float* bias,
                           const float* input, float* output,
                           size_t output_dim, size_t input_dim) {
    for (size_t i = 0; i < output_dim; ++i) {
        float sum = bias[i];
        for (size_t j = 0; j < input_dim; ++j) {
            sum += W[i * input_dim + j] * input[j];
        }
        output[i] = sum;
    }
}

/// ReLU activation (in-place)
static void relu_inplace(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

/// Softmax (in-place, numerically stable)
static void softmax_inplace(float* data, size_t n) {
    float max_val = *std::max_element(data, data + n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    for (size_t i = 0; i < n; ++i) {
        data[i] /= sum;
    }
}

/// Argmax: return index of maximum element
static size_t argmax(const float* data, size_t n) {
    return static_cast<size_t>(
        std::max_element(data, data + n) - data);
}

/// Run inference on a single image through the FC network.
/// Returns predicted digit (0-9).
static int inference_single(const model::Model& model, const float* image) {
    // Layer 1: Linear(784 → 128) + ReLU
    std::vector<float> hidden(model.layers[0].output_dim);
    linear_forward(model.layers[0].weights.data(),
                   model.layers[0].bias.data(),
                   image, hidden.data(),
                   model.layers[0].output_dim,
                   model.layers[0].input_dim);
    relu_inplace(hidden.data(), hidden.size());

    // Layer 2: Linear(128 → 10) + Softmax
    std::vector<float> logits(model.layers[1].output_dim);
    linear_forward(model.layers[1].weights.data(),
                   model.layers[1].bias.data(),
                   hidden.data(), logits.data(),
                   model.layers[1].output_dim,
                   model.layers[1].input_dim);
    softmax_inplace(logits.data(), logits.size());

    return static_cast<int>(argmax(logits.data(), logits.size()));
}

/// Run inference using the Arena allocator (zero malloc on hot path).
static int inference_arena(const model::Model& model, const float* image,
                           memory::Arena& arena) {
    arena.reset();

    size_t hidden_dim = model.layers[0].output_dim;
    size_t output_dim = model.layers[1].output_dim;

    // Allocate from arena — O(1), no malloc
    float* hidden = arena.alloc<float>(hidden_dim);
    float* logits = arena.alloc<float>(output_dim);

    // Layer 1: Linear + ReLU
    linear_forward(model.layers[0].weights.data(),
                   model.layers[0].bias.data(),
                   image, hidden,
                   model.layers[0].output_dim,
                   model.layers[0].input_dim);
    relu_inplace(hidden, hidden_dim);

    // Layer 2: Linear + Softmax
    linear_forward(model.layers[1].weights.data(),
                   model.layers[1].bias.data(),
                   hidden, logits,
                   model.layers[1].output_dim,
                   model.layers[1].input_dim);
    softmax_inplace(logits, output_dim);

    return static_cast<int>(argmax(logits, output_dim));
}

int main(int argc, char* argv[]) {
    std::string model_dir = "models";
    if (argc > 1) {
        model_dir = argv[1];
    }

    std::string weights_path = model_dir + "/mnist_fc_weights.bin";
    std::string images_path = model_dir + "/mnist_test_images.bin";
    std::string labels_path = model_dir + "/mnist_test_labels.bin";

    // ─── Load Model ──────────────────────────────────────────────────────────
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "  InferX MNIST Inference Engine\n";
    std::cout << "═══════════════════════════════════════════\n\n";

    model::Model model;
    try {
        model = model::load_model(weights_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        std::cerr << "\nTo generate weights, run:\n";
        std::cerr << "  python scripts/export_mnist_weights.py\n";
        std::cerr << "Or to generate test weights (no PyTorch needed):\n";
        std::cerr << "  ./build-release/examples/generate_test_weights " << model_dir << "\n";
        return 1;
    }

    model.print_summary();
    std::cout << "\n";

    // ─── Load Test Data ──────────────────────────────────────────────────────
    model::TestData test_data;
    try {
        test_data = model::load_test_data(images_path, labels_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading test data: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Test samples: " << test_data.images.size()
              << " (image size: " << test_data.image_size << ")\n\n";

    // ─── Run Inference ───────────────────────────────────────────────────────
    memory::Arena arena(64 * 1024); // 64 KB — more than enough for MNIST

    size_t correct = 0;
    size_t total = test_data.images.size();

    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < total; ++i) {
        int predicted = inference_arena(model, test_data.images[i].data(), arena);
        int ground_truth = static_cast<int>(test_data.labels[i]);

        if (predicted == ground_truth) {
            ++correct;
        }
    }

    auto end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double per_image_us = (total_ms * 1000.0) / static_cast<double>(total);

    // ─── Report Results ──────────────────────────────────────────────────────
    double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(total);

    std::cout << "─── Results ────────────────────────────────\n";
    std::cout << "  Accuracy: " << correct << "/" << total
              << " (" << accuracy << "%)\n";
    std::cout << "  Total time: " << total_ms << " ms\n";
    std::cout << "  Per-image: " << per_image_us << " µs\n";
    std::cout << "  Throughput: " << (1e6 / per_image_us) << " images/sec\n";
    std::cout << "  Arena peak: " << arena.peak_usage() << " bytes\n";
    std::cout << "────────────────────────────────────────────\n";

    // Show first few predictions
    std::cout << "\nFirst 10 predictions:\n";
    for (size_t i = 0; i < std::min(size_t(10), total); ++i) {
        int pred = inference_arena(model, test_data.images[i].data(), arena);
        int gt = static_cast<int>(test_data.labels[i]);
        std::cout << "  Image " << i << ": predicted=" << pred
                  << " actual=" << gt
                  << (pred == gt ? " ✓" : " ✗") << "\n";
    }

    return (accuracy >= 90.0) ? 0 : 1; // Exit 0 if accuracy is reasonable
}
