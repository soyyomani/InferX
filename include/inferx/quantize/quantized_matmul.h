#pragma once

/// @file quantized_matmul.h
/// @brief INT8 matrix multiplication with dequantized float32 output.
///
/// Quantized MatMul pipeline:
/// ──────────────────────────
///   1. Quantize A (float32 → int8) with scale_a
///   2. Quantize B (float32 → int8) with scale_b
///   3. Compute C_int32 = A_int8 × B_int8 (integer arithmetic, no FP!)
///   4. Dequantize: C_float = C_int32 * scale_a * scale_b
///
/// Why INT32 accumulator?
///   int8 × int8 = int16, but we sum K such products, so we need int32
///   to avoid overflow: max accumulation = K × 127 × 127 = K × 16129
///   For K=768 (GPT-2), max = 12,387,072 — fits in int32 (max 2.1B).
///
/// Performance benefits:
///   - 4× less memory bandwidth (int8 vs float32)
///   - ARM NEON: sdot instruction does 4× int8 multiplies + int32 accumulate
///     in ONE instruction (vs 4 fmul + 4 fadd for float32)
///   - x86: VNNI (vpdpbusd) does 4× int8→int32 in one cycle
///
/// Memory savings for common models:
///   - GPT-2 small: 500 MB → 125 MB (4× reduction)
///   - LLaMA-7B: 14 GB → 3.5 GB (fits in consumer GPU VRAM)
///   - ResNet-50: 100 MB → 25 MB (fits in edge device)
///
/// This is the same approach used by:
///   - TensorRT: IInt8Calibrator + QDQ nodes
///   - ONNX Runtime: QLinearMatMul operator
///   - llama.cpp: Q8_0 quantization format
///   - Apple Neural Engine: requires INT8 for hardware acceleration

#include <inferx/quantize/quantize.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>

namespace inferx::quantize {

/// Result of a quantized matmul operation.
struct QuantizedMatMulResult {
    std::vector<float> output;          ///< Dequantized float32 output [M × N]
    std::vector<int32_t> raw_int32;     ///< Raw int32 accumulator (before dequant)
    float output_scale;                 ///< Combined scale: scale_a * scale_b
    size_t M, N;                        ///< Output dimensions
};

/// INT8 matrix multiplication: C_float = dequant(A_int8 × B_int8)
///
/// @param A_q   Quantized matrix A [M × K] (int8)
/// @param B_q   Quantized matrix B [K × N] (int8)
/// @param M     Rows of A
/// @param K     Inner dimension
/// @param N     Columns of B
/// @param scale_a  Quantization scale of A
/// @param scale_b  Quantization scale of B
/// @return QuantizedMatMulResult with dequantized float output
[[nodiscard]] inline QuantizedMatMulResult quantized_matmul(
    const int8_t* A_q, const int8_t* B_q,
    size_t M, size_t K, size_t N,
    float scale_a, float scale_b) {

    QuantizedMatMulResult result;
    result.M = M;
    result.N = N;
    result.output_scale = scale_a * scale_b;
    result.raw_int32.resize(M * N, 0);
    result.output.resize(M * N);

    // Step 1: Integer matmul with int32 accumulation
    // This is where the compute savings happen:
    // int8 multiply is cheaper than float32, and we pack 4× more data
    // per cache line, reducing memory bandwidth pressure.
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            int32_t a_val = static_cast<int32_t>(A_q[i * K + k]);
            for (size_t j = 0; j < N; ++j) {
                int32_t b_val = static_cast<int32_t>(B_q[k * N + j]);
                result.raw_int32[i * N + j] += a_val * b_val;
            }
        }
    }

    // Step 2: Dequantize int32 accumulator to float32
    // C_float[i][j] = C_int32[i][j] * scale_a * scale_b
    float combined_scale = scale_a * scale_b;
    for (size_t i = 0; i < M * N; ++i) {
        result.output[i] = static_cast<float>(result.raw_int32[i]) * combined_scale;
    }

    return result;
}

/// End-to-end quantized matmul: float inputs → quantize → int8 matmul → dequantize → float output
///
/// This is the "dynamic quantization" pattern used in ONNX Runtime:
///   - Weights are pre-quantized (offline)
///   - Activations are quantized per-inference (dynamic range)
///   - Output is float32 for the next layer
///
/// @param A  Float32 matrix [M × K]
/// @param B  Float32 matrix [K × N]
/// @param M  Rows of A
/// @param K  Inner dimension
/// @param N  Columns of B
[[nodiscard]] inline QuantizedMatMulResult quantized_matmul_dynamic(
    const float* A, const float* B,
    size_t M, size_t K, size_t N) {

    // Quantize A
    QuantParams params_a = compute_symmetric_params(A, M * K);
    std::vector<int8_t> A_q(M * K);
    quantize_tensor(A, A_q.data(), M * K, params_a);

    // Quantize B
    QuantParams params_b = compute_symmetric_params(B, K * N);
    std::vector<int8_t> B_q(K * N);
    quantize_tensor(B, B_q.data(), K * N, params_b);

    // Perform int8 matmul + dequantize
    return quantized_matmul(A_q.data(), B_q.data(), M, K, N,
                            params_a.scale, params_b.scale);
}

/// Convenience: vector inputs
[[nodiscard]] inline QuantizedMatMulResult quantized_matmul_dynamic(
    const std::vector<float>& A, const std::vector<float>& B,
    size_t M, size_t K, size_t N) {
    return quantized_matmul_dynamic(A.data(), B.data(), M, K, N);
}

/// Compare quantized matmul accuracy against float32 matmul.
/// Returns the quantization error metrics.
[[nodiscard]] inline QuantError compare_with_float(
    const float* A, const float* B,
    size_t M, size_t K, size_t N) {

    // Float32 reference
    std::vector<float> C_fp32(M * N, 0.0f);
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_val = A[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                C_fp32[i * N + j] += a_val * B[k * N + j];
            }
        }
    }

    // Quantized result
    auto q_result = quantized_matmul_dynamic(A, B, M, K, N);

    // Measure error
    return measure_error(C_fp32.data(), q_result.output.data(), M * N);
}

} // namespace inferx::quantize
