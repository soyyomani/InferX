#pragma once

/// @file model_loader.h
/// @brief Binary model weight loader for InferX.
///
/// File format (.bin):
///   Bytes 0-3:   Magic "INFX" (4 bytes)
///   Bytes 4-7:   uint32_t num_layers
///   For each layer:
///     uint32_t rows (output_dim)
///     uint32_t cols (input_dim)
///     float[rows * cols] weight matrix (row-major)
///     float[rows] bias vector
///
/// Test data format:
///   Images: uint32_t num_samples, uint32_t image_size, then float[num*size]
///   Labels: uint32_t num_samples, then uint8_t[num]
///
/// This is intentionally simple — no protobuf, no ONNX, no external deps.
/// The point is to demonstrate model loading + inference, not format parsing.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace inferx::model {

/// A single fully-connected layer's parameters.
struct FCLayer {
    size_t input_dim;               ///< Number of input features (cols)
    size_t output_dim;              ///< Number of output features (rows)
    std::vector<float> weights;     ///< Weight matrix [output_dim × input_dim] row-major
    std::vector<float> bias;        ///< Bias vector [output_dim]
};

/// A loaded model: sequence of FC layers.
struct Model {
    std::vector<FCLayer> layers;
    std::string name;

    /// Total parameter count
    [[nodiscard]] size_t total_params() const noexcept {
        size_t total = 0;
        for (const auto& l : layers) {
            total += l.weights.size() + l.bias.size();
        }
        return total;
    }

    /// Total memory for weights (bytes)
    [[nodiscard]] size_t weight_memory_bytes() const noexcept {
        return total_params() * sizeof(float);
    }

    /// Print model summary
    void print_summary() const {
        std::cout << "Model: " << name << "\n";
        std::cout << "Layers: " << layers.size() << "\n";
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& l = layers[i];
            std::cout << "  Layer " << i << ": Linear("
                      << l.input_dim << " → " << l.output_dim << ") "
                      << "[" << l.weights.size() + l.bias.size() << " params]\n";
        }
        std::cout << "Total params: " << total_params()
                  << " (" << weight_memory_bytes() / 1024 << " KB)\n";
    }
};

/// Load model weights from InferX binary format.
/// @param path  Path to .bin file
/// @return Loaded model with all layer weights
[[nodiscard]] inline Model load_model(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + path);
    }

    // Read magic header
    char magic[4];
    file.read(magic, 4);
    if (std::strncmp(magic, "INFX", 4) != 0) {
        throw std::runtime_error("Invalid model file: bad magic header");
    }

    // Read layer count
    uint32_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));

    Model model;
    model.name = path;
    model.layers.resize(num_layers);

    for (uint32_t i = 0; i < num_layers; ++i) {
        uint32_t rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

        auto& layer = model.layers[i];
        layer.output_dim = rows;
        layer.input_dim = cols;

        // Read weight matrix
        layer.weights.resize(rows * cols);
        file.read(reinterpret_cast<char*>(layer.weights.data()),
                  rows * cols * sizeof(float));

        // Read bias vector
        layer.bias.resize(rows);
        file.read(reinterpret_cast<char*>(layer.bias.data()),
                  rows * sizeof(float));

        if (!file.good()) {
            throw std::runtime_error("Error reading layer " + std::to_string(i));
        }
    }

    return model;
}

/// Test image data loaded from binary.
struct TestData {
    std::vector<std::vector<float>> images;  ///< Each image is 784 floats
    std::vector<uint8_t> labels;             ///< Ground truth labels
    size_t image_size = 0;                   ///< Pixels per image (784)
};

/// Load test images and labels.
/// @param images_path  Path to mnist_test_images.bin
/// @param labels_path  Path to mnist_test_labels.bin
[[nodiscard]] inline TestData load_test_data(const std::string& images_path,
                                             const std::string& labels_path) {
    TestData data;

    // Load images
    {
        std::ifstream f(images_path, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open: " + images_path);
        }

        uint32_t num_samples, img_size;
        f.read(reinterpret_cast<char*>(&num_samples), sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&img_size), sizeof(uint32_t));

        data.image_size = img_size;
        data.images.resize(num_samples);

        for (uint32_t i = 0; i < num_samples; ++i) {
            data.images[i].resize(img_size);
            f.read(reinterpret_cast<char*>(data.images[i].data()),
                   img_size * sizeof(float));
        }
    }

    // Load labels
    {
        std::ifstream f(labels_path, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open: " + labels_path);
        }

        uint32_t num_samples;
        f.read(reinterpret_cast<char*>(&num_samples), sizeof(uint32_t));

        data.labels.resize(num_samples);
        f.read(reinterpret_cast<char*>(data.labels.data()), num_samples);
    }

    return data;
}

/// Save model weights in InferX binary format (for generating test weights).
inline void save_model(const Model& model, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create model file: " + path);
    }

    // Magic header
    file.write("INFX", 4);

    // Layer count
    uint32_t num_layers = static_cast<uint32_t>(model.layers.size());
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(uint32_t));

    for (const auto& layer : model.layers) {
        uint32_t rows = static_cast<uint32_t>(layer.output_dim);
        uint32_t cols = static_cast<uint32_t>(layer.input_dim);
        file.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(layer.weights.data()),
                   layer.weights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(layer.bias.data()),
                   layer.bias.size() * sizeof(float));
    }
}

/// Save test data in binary format.
inline void save_test_data(const TestData& data,
                           const std::string& images_path,
                           const std::string& labels_path) {
    // Save images
    {
        std::ofstream f(images_path, std::ios::binary);
        uint32_t num = static_cast<uint32_t>(data.images.size());
        uint32_t size = static_cast<uint32_t>(data.image_size);
        f.write(reinterpret_cast<const char*>(&num), sizeof(uint32_t));
        f.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
        for (const auto& img : data.images) {
            f.write(reinterpret_cast<const char*>(img.data()),
                    img.size() * sizeof(float));
        }
    }

    // Save labels
    {
        std::ofstream f(labels_path, std::ios::binary);
        uint32_t num = static_cast<uint32_t>(data.labels.size());
        f.write(reinterpret_cast<const char*>(&num), sizeof(uint32_t));
        f.write(reinterpret_cast<const char*>(data.labels.data()), data.labels.size());
    }
}

} // namespace inferx::model
