/// @file generate_test_weights.cpp
/// @brief Generate deterministic test weights for MNIST inference without PyTorch.
///
/// This creates synthetic model weights and test data so the C++ inference
/// example can be built and run without any Python/PyTorch dependency.
///
/// The weights are NOT trained — they use a simple hand-crafted pattern:
///   Layer 1: Each of 128 hidden neurons "looks at" a different region of the image
///   Layer 2: Each of 10 output neurons corresponds to a digit class
///
/// With these synthetic weights, accuracy will be low (~10-20%), but the
/// pipeline is fully exercised: load → forward → softmax → argmax.
///
/// For real accuracy (97%+), run: python scripts/export_mnist_weights.py
///
/// Usage:
///   ./generate_test_weights [output_dir]
///   ./generate_test_weights models/

#include <inferx/model/model_loader.h>

#include <iostream>
#include <random>
#include <cmath>
#include <string>

using namespace inferx::model;

int main(int argc, char* argv[]) {
    std::string output_dir = "models";
    if (argc > 1) {
        output_dir = argv[1];
    }

    std::cout << "Generating synthetic MNIST test weights...\n";

    // ─── Generate Model Weights ──────────────────────────────────────────────
    Model model;
    model.name = "mnist_synthetic";

    // Use Xavier initialization (reasonable random weights)
    std::mt19937 gen(42); // Deterministic seed

    // Layer 1: Linear(784, 128)
    {
        FCLayer layer;
        layer.input_dim = 784;
        layer.output_dim = 128;

        float std_dev = std::sqrt(2.0f / (784.0f + 128.0f)); // Xavier
        std::normal_distribution<float> dist(0.0f, std_dev);

        layer.weights.resize(128 * 784);
        for (auto& w : layer.weights) w = dist(gen);

        layer.bias.resize(128, 0.0f);
        model.layers.push_back(std::move(layer));
    }

    // Layer 2: Linear(128, 10)
    {
        FCLayer layer;
        layer.input_dim = 128;
        layer.output_dim = 10;

        float std_dev = std::sqrt(2.0f / (128.0f + 10.0f)); // Xavier
        std::normal_distribution<float> dist(0.0f, std_dev);

        layer.weights.resize(10 * 128);
        for (auto& w : layer.weights) w = dist(gen);

        layer.bias.resize(10, 0.0f);
        model.layers.push_back(std::move(layer));
    }

    // Save model
    std::string weights_path = output_dir + "/mnist_fc_weights.bin";
    save_model(model, weights_path);
    model.print_summary();

    // ─── Generate Test Images ────────────────────────────────────────────────
    // Create 100 synthetic "digit" images:
    // Each image has a bright region corresponding to its label
    TestData data;
    data.image_size = 784; // 28×28
    const size_t num_samples = 100;

    std::uniform_real_distribution<float> noise(-0.1f, 0.1f);

    for (size_t i = 0; i < num_samples; ++i) {
        std::vector<float> image(784, 0.0f);

        // Label cycles through 0-9
        uint8_t label = static_cast<uint8_t>(i % 10);

        // Create a simple pattern: bright pixels in a region that
        // varies by digit (so a trained model could learn this)
        // Digit k → bright pixels in rows [k*2, k*2+4], cols [k*2, k*2+8]
        int row_start = label * 2;
        int col_start = label * 2;
        for (int r = row_start; r < std::min(row_start + 5, 28); ++r) {
            for (int c = col_start; c < std::min(col_start + 8, 28); ++c) {
                image[r * 28 + c] = 2.0f + noise(gen); // Normalized value (like MNIST after norm)
            }
        }

        // Add some background noise
        for (auto& px : image) {
            px += noise(gen) * 0.1f;
        }

        data.images.push_back(std::move(image));
        data.labels.push_back(label);
    }

    // Save test data
    std::string images_path = output_dir + "/mnist_test_images.bin";
    std::string labels_path = output_dir + "/mnist_test_labels.bin";
    save_test_data(data, images_path, labels_path);

    std::cout << "\nGenerated:\n";
    std::cout << "  " << weights_path << " (" << model.weight_memory_bytes() / 1024 << " KB)\n";
    std::cout << "  " << images_path << " (" << num_samples << " images)\n";
    std::cout << "  " << labels_path << " (" << num_samples << " labels)\n";
    std::cout << "\nNote: These are SYNTHETIC weights (not trained).\n";
    std::cout << "For real accuracy, run: python scripts/export_mnist_weights.py\n";

    return 0;
}
