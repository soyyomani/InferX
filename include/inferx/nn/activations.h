#pragma once

#include <inferx/core/tracer.h>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace inferx::nn {

/// Activation functions with step-by-step tracing for education.
class Activations {
public:
    /// ReLU: max(0, x)
    static std::vector<float> relu(const std::vector<float>& input) {
        auto& tracer = core::Tracer::instance();
        int n = static_cast<int>(input.size());

        tracer.record("ReLU", "ReLU: Rectified Linear Unit",
            "f(x) = max(0, x) — The most popular activation function",
            {"Formula: output[i] = max(0, input[i])",
             "If positive → keep it. If negative → set to 0.",
             "",
             "Why ReLU?",
             "  • Simple and fast (just a comparison)",
             "  • No vanishing gradient for positive values",
             "  • Introduces non-linearity (networks can learn curves)",
             "  • Without activation: stacking linear layers = one linear layer"});

        std::vector<float> output(n);
        int zeros = 0;
        for (int i = 0; i < n; i++) {
            output[i] = std::max(0.0f, input[i]);
            if (output[i] == 0) zeros++;
        }

        std::vector<std::string> details;
        for (int i = 0; i < std::min(n, 6); i++) {
            std::ostringstream oss;
            oss << "max(0, " << std::fixed << std::setprecision(4)
                << input[i] << ") = " << output[i];
            if (input[i] < 0) oss << "  ← zeroed (was negative)";
            details.push_back(oss.str());
        }
        if (n > 6) details.push_back("... (" + std::to_string(n - 6) + " more)");
        details.push_back("");
        details.push_back("Sparsity: " + std::to_string(zeros) + "/" +
            std::to_string(n) + " neurons dead (" +
            std::to_string(zeros * 100 / n) + "%)");

        tracer.record("ReLU", "ReLU Applied",
            std::to_string(zeros) + " of " + std::to_string(n) + " values zeroed out",
            details);

        return output;
    }

    /// GELU: Gaussian Error Linear Unit (used in GPT, BERT)
    static std::vector<float> gelu(const std::vector<float>& input) {
        auto& tracer = core::Tracer::instance();
        int n = static_cast<int>(input.size());

        tracer.record("GELU", "GELU: Gaussian Error Linear Unit",
            "f(x) = x × Φ(x) where Φ is the CDF of standard normal",
            {"Formula: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))",
             "",
             "Why GELU over ReLU?",
             "  • Smoother: doesn't have a hard kink at 0",
             "  • Small negative values aren't completely zeroed",
             "  • Used in GPT-2, GPT-3, BERT, most modern transformers",
             "  • Better gradient flow during training",
             "",
             "Intuition: soft gate that's ~0 for very negative, ~x for positive"});

        std::vector<float> output(n);
        std::vector<std::string> details;
        const float sqrt_2_over_pi = std::sqrt(2.0f / 3.14159265f);

        for (int i = 0; i < n; i++) {
            float x = input[i];
            float inner = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
            float tanh_val = std::tanh(inner);
            output[i] = 0.5f * x * (1.0f + tanh_val);

            if (i < 4) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4);
                oss << "GELU(" << x << ") = 0.5×" << x << "×(1+tanh("
                    << inner << ")) = " << output[i];
                details.push_back(oss.str());
            }
        }
        if (n > 4) details.push_back("... (" + std::to_string(n - 4) + " more)");

        tracer.record("GELU", "GELU Applied",
            "Smooth non-linearity applied to all values",
            details);

        return output;
    }

    /// LayerNorm: normalize then scale+shift
    static std::vector<float> layer_norm(
        const std::vector<float>& input, int dim) {
        auto& tracer = core::Tracer::instance();
        int n = static_cast<int>(input.size());
        int num_vecs = n / dim;

        tracer.record("LayerNorm", "Layer Normalization",
            "Normalize each vector to mean=0, std=1, then apply learned scale+shift",
            {"Input: " + std::to_string(num_vecs) + " vectors of dim " + std::to_string(dim),
             "For each vector independently:",
             "  1. Compute mean: μ = (1/d) × Σx_i",
             "  2. Compute variance: σ² = (1/d) × Σ(x_i - μ)²",
             "  3. Normalize: x̂_i = (x_i - μ) / √(σ² + ε)",
             "  4. Scale + shift: y_i = γ × x̂_i + β",
             "",
             "Why? Keeps values in a stable range as they flow through layers."});

        std::vector<float> output(n);
        const float eps = 1e-5f;

        for (int v = 0; v < num_vecs; v++) {
            int offset = v * dim;
            float mean = 0, var = 0;

            for (int i = 0; i < dim; i++)
                mean += input[offset + i];
            mean /= dim;

            for (int i = 0; i < dim; i++) {
                float diff = input[offset + i] - mean;
                var += diff * diff;
            }
            var /= dim;

            float inv_std = 1.0f / std::sqrt(var + eps);
            for (int i = 0; i < dim; i++) {
                output[offset + i] = (input[offset + i] - mean) * inv_std;
            }

            if (v < 2) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4);
                oss << "vec[" << v << "]: μ=" << mean << ", σ²=" << var
                    << ", 1/√(σ²+ε)=" << inv_std;
                tracer.record("LayerNorm",
                    "Normalizing vector " + std::to_string(v),
                    oss.str(), {});
            }
        }

        return output;
    }
};

} // namespace inferx::nn
