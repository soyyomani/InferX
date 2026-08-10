#pragma once

/// @file quantize.h
/// @brief INT8 quantization primitives for inference optimization.
///
/// Why quantization?
/// ─────────────────
/// Neural network weights and activations are typically stored as float32 (4 bytes).
/// Quantization maps these to int8 (1 byte), achieving:
///   - 4× memory reduction (critical for mobile/edge deployment)
///   - 2-4× compute speedup (INT8 instructions are faster + more fit in SIMD)
///   - Minimal accuracy loss (<1% for well-calibrated models)
///
/// Quantization math (per-tensor symmetric):
///   scale = max(|tensor|) / 127.0
///   quantized[i] = clamp(round(tensor[i] / scale), -128, 127)
///   dequantized[i] = quantized[i] * scale
///
/// Symmetric vs Asymmetric:
///   - Symmetric: zero_point = 0, range = [-127, 127], simpler math
///   - Asymmetric: zero_point ≠ 0, range = [-128, 127], better for ReLU outputs
///   We implement both, but symmetric is the default (used by TensorRT, oneDNN).
///
/// Quantization schemes:
///   - Per-tensor: one scale for entire tensor (fastest, least accurate)
///   - Per-channel: one scale per output channel (good accuracy/speed trade-off)
///   - Per-token: one scale per sequence position (used in LLM quantization)
///
/// Real-world usage:
///   - TensorRT: INT8 calibration → per-tensor/per-channel quantization
///   - CoreML: palettization + per-channel quantization
///   - ONNX Runtime: QLinearMatMul, DynamicQuantizeLinear
///   - llama.cpp: GGML Q4/Q8 quantization for LLM inference

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <stdexcept>

namespace inferx::quantize {

// ─── Quantization Parameters ─────────────────────────────────────────────────

/// Quantization parameters for a tensor.
/// These define the mapping between float32 and int8 domains.
struct QuantParams {
    float scale = 1.0f;        ///< Scale factor: float_value = int_value * scale
    int32_t zero_point = 0;    ///< Zero point offset (0 for symmetric)

    /// Quantize a single float value to int8
    [[nodiscard]] int8_t quantize(float value) const noexcept {
        float scaled = value / scale + static_cast<float>(zero_point);
        int32_t rounded = static_cast<int32_t>(std::round(scaled));
        return static_cast<int8_t>(std::clamp(rounded, -128, 127));
    }

    /// Dequantize a single int8 value back to float
    [[nodiscard]] float dequantize(int8_t value) const noexcept {
        return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
    }
};

// ─── Scale Computation ───────────────────────────────────────────────────────

/// Compute symmetric quantization parameters.
/// Maps [-max_abs, +max_abs] → [-127, +127]
/// Zero is mapped exactly to zero (important for ReLU outputs with many zeros).
inline QuantParams compute_symmetric_params(const float* data, size_t n) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        max_abs = std::max(max_abs, std::abs(data[i]));
    }

    QuantParams params;
    params.scale = max_abs / 127.0f;
    params.zero_point = 0;

    // Avoid division by zero for all-zero tensors
    if (params.scale == 0.0f) {
        params.scale = 1.0f;
    }

    return params;
}

/// Compute asymmetric quantization parameters.
/// Maps [min, max] → [-128, 127]
/// Better utilization of the int8 range for non-symmetric distributions
/// (e.g., ReLU outputs which are always >= 0).
inline QuantParams compute_asymmetric_params(const float* data, size_t n) {
    float min_val = data[0], max_val = data[0];
    for (size_t i = 1; i < n; ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }

    QuantParams params;
    params.scale = (max_val - min_val) / 255.0f;
    if (params.scale == 0.0f) params.scale = 1.0f;
    params.zero_point = static_cast<int32_t>(std::round(-min_val / params.scale)) - 128;

    return params;
}

// ─── Bulk Quantization / Dequantization ──────────────────────────────────────

/// Quantize a float32 tensor to int8 (bulk operation).
/// @param input   Float32 source data
/// @param output  Int8 destination (must be pre-allocated, size n)
/// @param n       Number of elements
/// @param params  Quantization parameters (from compute_*_params)
inline void quantize_tensor(const float* input, int8_t* output, size_t n,
                            const QuantParams& params) {
    const float inv_scale = 1.0f / params.scale;
    const float zp = static_cast<float>(params.zero_point);

    for (size_t i = 0; i < n; ++i) {
        float scaled = input[i] * inv_scale + zp;
        int32_t rounded = static_cast<int32_t>(std::round(scaled));
        output[i] = static_cast<int8_t>(std::clamp(rounded, -128, 127));
    }
}

/// Dequantize an int8 tensor back to float32 (bulk operation).
/// @param input   Int8 source data
/// @param output  Float32 destination (must be pre-allocated, size n)
/// @param n       Number of elements
/// @param params  Quantization parameters
inline void dequantize_tensor(const int8_t* input, float* output, size_t n,
                              const QuantParams& params) {
    const float zp = static_cast<float>(params.zero_point);

    for (size_t i = 0; i < n; ++i) {
        output[i] = (static_cast<float>(input[i]) - zp) * params.scale;
    }
}

// ─── Convenience Wrappers ────────────────────────────────────────────────────

/// Quantize a vector, returning quantized data + params.
struct QuantizedTensor {
    std::vector<int8_t> data;
    QuantParams params;
    std::vector<size_t> shape;

    [[nodiscard]] size_t numel() const noexcept {
        if (shape.empty()) return data.size();
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    /// Dequantize back to float32
    [[nodiscard]] std::vector<float> dequantize() const {
        std::vector<float> result(data.size());
        dequantize_tensor(data.data(), result.data(), data.size(), params);
        return result;
    }

    /// Memory savings: ratio of int8 size to float32 size
    [[nodiscard]] double compression_ratio() const noexcept {
        return 4.0; // float32 (4 bytes) → int8 (1 byte) = 4× compression
    }
};

/// Quantize a float vector with symmetric quantization (most common).
[[nodiscard]] inline QuantizedTensor quantize_symmetric(
    const std::vector<float>& input, std::vector<size_t> shape = {}) {
    QuantizedTensor result;
    result.params = compute_symmetric_params(input.data(), input.size());
    result.data.resize(input.size());
    quantize_tensor(input.data(), result.data.data(), input.size(), result.params);
    result.shape = shape.empty() ? std::vector<size_t>{input.size()} : std::move(shape);
    return result;
}

/// Quantize with asymmetric quantization (better for ReLU outputs).
[[nodiscard]] inline QuantizedTensor quantize_asymmetric(
    const std::vector<float>& input, std::vector<size_t> shape = {}) {
    QuantizedTensor result;
    result.params = compute_asymmetric_params(input.data(), input.size());
    result.data.resize(input.size());
    quantize_tensor(input.data(), result.data.data(), input.size(), result.params);
    result.shape = shape.empty() ? std::vector<size_t>{input.size()} : std::move(shape);
    return result;
}

// ─── Error Metrics ───────────────────────────────────────────────────────────

/// Compute quantization error metrics (useful for calibration evaluation).
struct QuantError {
    float max_abs_error = 0.0f;     ///< Maximum absolute error
    float mean_abs_error = 0.0f;    ///< Mean absolute error
    float rmse = 0.0f;             ///< Root mean squared error
    float snr_db = 0.0f;           ///< Signal-to-noise ratio in dB
};

/// Measure quantization error between original and dequantized values.
[[nodiscard]] inline QuantError measure_error(const float* original,
                                              const float* dequantized,
                                              size_t n) {
    QuantError err;
    float sum_abs = 0.0f;
    float sum_sq_error = 0.0f;
    float sum_sq_signal = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(original[i] - dequantized[i]);
        err.max_abs_error = std::max(err.max_abs_error, diff);
        sum_abs += diff;
        sum_sq_error += diff * diff;
        sum_sq_signal += original[i] * original[i];
    }

    err.mean_abs_error = sum_abs / static_cast<float>(n);
    err.rmse = std::sqrt(sum_sq_error / static_cast<float>(n));

    if (sum_sq_error > 0.0f && sum_sq_signal > 0.0f) {
        err.snr_db = 10.0f * std::log10(sum_sq_signal / sum_sq_error);
    }

    return err;
}

} // namespace inferx::quantize
