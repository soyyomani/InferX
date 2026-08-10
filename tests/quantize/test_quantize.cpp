/// @file test_quantize.cpp
/// @brief Tests for INT8 quantization primitives and quantized matmul.

#include <gtest/gtest.h>
#include <inferx/quantize/quantize.h>
#include <inferx/quantize/quantized_matmul.h>

#include <vector>
#include <random>
#include <cmath>
#include <numeric>

using namespace inferx::quantize;

// ═══════════════════════════════════════════════════════════════════════════════
// Quantization Parameter Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(QuantParams, SymmetricBasic) {
    std::vector<float> data = {-1.0f, 0.0f, 0.5f, 1.0f};
    auto params = compute_symmetric_params(data.data(), data.size());
    EXPECT_FLOAT_EQ(params.scale, 1.0f / 127.0f);
    EXPECT_EQ(params.zero_point, 0);
}

TEST(QuantParams, SymmetricZeroMapsToZero) {
    std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto params = compute_symmetric_params(data.data(), data.size());
    // Zero must map exactly to 0 in symmetric quantization
    EXPECT_EQ(params.quantize(0.0f), 0);
}

TEST(QuantParams, SymmetricMaxMapsTo127) {
    std::vector<float> data = {-3.0f, 0.0f, 3.0f};
    auto params = compute_symmetric_params(data.data(), data.size());
    EXPECT_EQ(params.quantize(3.0f), 127);
    EXPECT_EQ(params.quantize(-3.0f), -127);
}

TEST(QuantParams, RoundTrip) {
    QuantParams params;
    params.scale = 0.1f;
    params.zero_point = 0;
    // Quantize then dequantize should be close to original
    float original = 0.5f;
    int8_t q = params.quantize(original);
    float deq = params.dequantize(q);
    EXPECT_NEAR(deq, original, params.scale); // Error bounded by scale
}

TEST(QuantParams, Clamping) {
    QuantParams params;
    params.scale = 0.01f; // Very fine scale
    params.zero_point = 0;
    // Value too large: should clamp to 127
    EXPECT_EQ(params.quantize(100.0f), 127);
    // Value too negative: should clamp to -128
    EXPECT_EQ(params.quantize(-100.0f), -128);
}

TEST(QuantParams, AllZeroTensor) {
    std::vector<float> data(100, 0.0f);
    auto params = compute_symmetric_params(data.data(), data.size());
    // Should not crash, scale should be non-zero
    EXPECT_GT(params.scale, 0.0f);
    EXPECT_EQ(params.quantize(0.0f), 0);
}

TEST(QuantParams, AsymmetricReLUOutput) {
    // ReLU output: all values >= 0
    std::vector<float> data = {0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    auto params = compute_asymmetric_params(data.data(), data.size());
    // Asymmetric should use full range [0, 3] → [-128, 127]
    EXPECT_GT(params.scale, 0.0f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bulk Quantize/Dequantize Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(BulkQuantize, RoundTripAccuracy) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> original(1000);
    for (auto& v : original) v = dist(gen);

    auto qt = quantize_symmetric(original);
    auto reconstructed = qt.dequantize();

    // Max error should be bounded by scale/2 (rounding error)
    float max_err = 0.0f;
    for (size_t i = 0; i < original.size(); ++i) {
        max_err = std::max(max_err, std::abs(original[i] - reconstructed[i]));
    }
    EXPECT_LT(max_err, qt.params.scale);
}

TEST(BulkQuantize, CompressionRatio) {
    std::vector<float> data(1024, 1.0f);
    auto qt = quantize_symmetric(data);
    EXPECT_DOUBLE_EQ(qt.compression_ratio(), 4.0);
    // Verify actual sizes
    EXPECT_EQ(qt.data.size(), 1024u);         // 1024 bytes (int8)
    EXPECT_EQ(data.size() * 4, 4096u);        // 4096 bytes (float32)
}

TEST(BulkQuantize, PreservesShape) {
    std::vector<float> data(24, 1.0f);
    auto qt = quantize_symmetric(data, {4, 6});
    EXPECT_EQ(qt.shape, (std::vector<size_t>{4, 6}));
    EXPECT_EQ(qt.numel(), 24u);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Error Metrics Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(QuantError, PerfectReconstruction) {
    // Quantize values that map exactly to int8
    QuantParams params;
    params.scale = 1.0f;
    params.zero_point = 0;
    std::vector<float> original = {-5, -3, 0, 3, 5};
    std::vector<float> reconstructed = original; // Perfect match

    auto err = measure_error(original.data(), reconstructed.data(), 5);
    EXPECT_FLOAT_EQ(err.max_abs_error, 0.0f);
    EXPECT_FLOAT_EQ(err.mean_abs_error, 0.0f);
    EXPECT_FLOAT_EQ(err.rmse, 0.0f);
}

TEST(QuantError, SNRForNormalDistribution) {
    // For normally distributed data, INT8 symmetric quantization
    // should achieve > 35 dB SNR (typical for well-calibrated models)
    std::mt19937 gen(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> original(10000);
    for (auto& v : original) v = dist(gen);

    auto qt = quantize_symmetric(original);
    auto reconstructed = qt.dequantize();

    auto err = measure_error(original.data(), reconstructed.data(), original.size());
    EXPECT_GT(err.snr_db, 35.0f); // > 35 dB is good quantization quality
}

// ═══════════════════════════════════════════════════════════════════════════════
// Quantized MatMul Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(QuantizedMatMul, KnownResult2x2) {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // FP32 result: [[19, 22], [43, 50]]
    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {5, 6, 7, 8};

    auto result = quantized_matmul_dynamic(A, B, 2, 2, 2);

    // Quantized result should be close to FP32
    EXPECT_NEAR(result.output[0], 19.0f, 1.0f);
    EXPECT_NEAR(result.output[1], 22.0f, 1.0f);
    EXPECT_NEAR(result.output[2], 43.0f, 1.5f);
    EXPECT_NEAR(result.output[3], 50.0f, 1.5f);
}

TEST(QuantizedMatMul, IdentityMatrix) {
    // A × I = A (with quantization error)
    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f}; // 2×2
    std::vector<float> I = {1.0f, 0.0f, 0.0f, 1.0f}; // 2×2 identity

    auto result = quantized_matmul_dynamic(A, I, 2, 2, 2);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(result.output[i], A[i], 0.5f);
    }
}

TEST(QuantizedMatMul, LargerMatrixAccuracy) {
    // Random 32×64 × 64×32 — check error is reasonable
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const size_t M = 32, K = 64, N = 32;
    std::vector<float> A(M * K), B(K * N);
    for (auto& v : A) v = dist(gen);
    for (auto& v : B) v = dist(gen);

    auto err = compare_with_float(A.data(), B.data(), M, K, N);

    // For random normal data, quantized matmul should achieve < 5% relative error
    // SNR should be > 20 dB for reasonable quantization
    EXPECT_GT(err.snr_db, 20.0f);
}

TEST(QuantizedMatMul, MLShapeGPT2FFN) {
    // GPT-2 FFN shape: [32, 768] × [768, 3072]
    // Use small values to simulate trained weight distribution
    std::mt19937 gen(99);
    std::normal_distribution<float> dist(0.0f, 0.02f); // Typical weight init

    const size_t M = 8, K = 64, N = 256; // Scaled down for test speed
    std::vector<float> A(M * K), B(K * N);
    for (auto& v : A) v = dist(gen);
    for (auto& v : B) v = dist(gen);

    auto err = compare_with_float(A.data(), B.data(), M, K, N);

    // With small normally-distributed values, quantization should work well
    EXPECT_GT(err.snr_db, 15.0f);
}

TEST(QuantizedMatMul, RawInt8MatMul) {
    // Direct int8 matmul test (no float conversion)
    // [1, 2] × [3, 4] = [1*3+2*5, 1*4+2*6] = [13, 16]
    // [3, 4]   [5, 6]   [3*3+4*5, 3*4+4*6]   [29, 36]
    std::vector<int8_t> A = {1, 2, 3, 4};
    std::vector<int8_t> B = {3, 4, 5, 6};

    auto result = quantized_matmul(A.data(), B.data(), 2, 2, 2, 1.0f, 1.0f);

    // With scale=1.0, output should exactly match int32 result
    EXPECT_EQ(result.raw_int32[0], 13);
    EXPECT_EQ(result.raw_int32[1], 16);
    EXPECT_EQ(result.raw_int32[2], 29);
    EXPECT_EQ(result.raw_int32[3], 36);
}

TEST(QuantizedMatMul, ScaleAppliedCorrectly) {
    std::vector<int8_t> A = {10, 20};  // 1×2
    std::vector<int8_t> B = {30, 40};  // 2×1
    float scale_a = 0.1f, scale_b = 0.2f;

    auto result = quantized_matmul(A.data(), B.data(), 1, 2, 1, scale_a, scale_b);

    // int32: 10*30 + 20*40 = 300 + 800 = 1100
    EXPECT_EQ(result.raw_int32[0], 1100);
    // dequant: 1100 * 0.1 * 0.2 = 22.0
    EXPECT_FLOAT_EQ(result.output[0], 22.0f);
}

TEST(QuantizedMatMul, OutputDimensions) {
    const size_t M = 4, K = 8, N = 16;
    std::vector<int8_t> A(M * K, 1);
    std::vector<int8_t> B(K * N, 1);

    auto result = quantized_matmul(A.data(), B.data(), M, K, N, 1.0f, 1.0f);

    EXPECT_EQ(result.output.size(), M * N);
    EXPECT_EQ(result.M, M);
    EXPECT_EQ(result.N, N);
    // Each element = sum of K ones × ones = K = 8
    for (auto v : result.raw_int32) {
        EXPECT_EQ(v, static_cast<int32_t>(K));
    }
}
