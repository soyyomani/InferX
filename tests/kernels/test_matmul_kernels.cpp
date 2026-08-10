/// @file test_matmul_kernels.cpp
/// @brief Correctness tests for all matmul kernel implementations.
///
/// Strategy: Use the naive kernel as the "reference" implementation (it's
/// obviously correct from the triple-loop structure), then verify that
/// optimized kernels produce the same results within floating-point tolerance.
///
/// Tests cover:
///   - Identity cases (multiplying by identity matrix)
///   - Known small matrices (hand-computed results)
///   - Random matrices at various sizes (cross-validated against naive)
///   - Edge cases: non-square, single row/col, remainder handling
///   - Dispatch heuristic correctness

#include <gtest/gtest.h>
#include <inferx/kernels/matmul_dispatch.h>

#include <vector>
#include <random>
#include <cmath>
#include <numeric>

using namespace inferx::kernels;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Generate random matrix with values in [-1, 1]
static std::vector<float> random_matrix(size_t rows, size_t cols, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> m(rows * cols);
    for (auto& v : m) v = dist(gen);
    return m;
}

/// Generate identity matrix [N × N]
static std::vector<float> identity_matrix(size_t N) {
    std::vector<float> m(N * N, 0.0f);
    for (size_t i = 0; i < N; ++i) m[i * N + i] = 1.0f;
    return m;
}

/// Check two matrices are element-wise close (relative + absolute tolerance)
static void expect_matrices_close(const std::vector<float>& expected,
                                   const std::vector<float>& actual,
                                   size_t M, size_t N,
                                   float atol = 1e-4f, float rtol = 1e-4f) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < M * N; ++i) {
        float diff = std::abs(expected[i] - actual[i]);
        float scale = std::max(std::abs(expected[i]), std::abs(actual[i]));
        EXPECT_LE(diff, atol + rtol * scale)
            << "Mismatch at index " << i << " (row=" << i / N << ", col=" << i % N << "): "
            << "expected=" << expected[i] << " actual=" << actual[i]
            << " diff=" << diff;
    }
}

// ─── Basic Correctness ───────────────────────────────────────────────────────

TEST(MatMulKernels, NaiveIdentity) {
    // A × I = A
    const size_t N = 4;
    std::vector<float> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto I = identity_matrix(N);
    auto C = matmul_naive(A, I, N, N, N);
    expect_matrices_close(A, C, N, N);
}

TEST(MatMulKernels, NaiveKnownResult) {
    // [1 2] × [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {5, 6, 7, 8};
    auto C = matmul_naive(A, B, 2, 2, 2);
    std::vector<float> expected = {19, 22, 43, 50};
    expect_matrices_close(expected, C, 2, 2, 1e-6f);
}

TEST(MatMulKernels, NaiveNonSquare) {
    // [1 2 3] × [1 2]   [1*1+2*3+3*5  1*2+2*4+3*6] = [22 28]
    // [4 5 6]   [3 4] = [4*1+5*3+6*5  4*2+5*4+6*6]   [49 64]
    //            [5 6]
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {1, 2, 3, 4, 5, 6};
    auto C = matmul_naive(A, B, 2, 3, 2);
    std::vector<float> expected = {22, 28, 49, 64};
    expect_matrices_close(expected, C, 2, 2, 1e-6f);
}

// ─── Cross-Validation: Tiled vs Naive ────────────────────────────────────────

class MatMulCrossValidation : public ::testing::TestWithParam<size_t> {};

TEST_P(MatMulCrossValidation, TiledMatchesNaive) {
    const size_t N = GetParam();
    auto A = random_matrix(N, N, 100);
    auto B = random_matrix(N, N, 200);

    auto C_naive = matmul_naive(A, B, N, N, N);
    auto C_tiled = matmul_tiled(A, B, N, N, N);

    // Larger matrices accumulate more FP error, so scale tolerance
    float tol = 1e-4f * static_cast<float>(N);
    expect_matrices_close(C_naive, C_tiled, N, N, tol, 1e-4f);
}

TEST_P(MatMulCrossValidation, NeonMatchesNaive) {
    const size_t N = GetParam();
    auto A = random_matrix(N, N, 300);
    auto B = random_matrix(N, N, 400);

    auto C_naive = matmul_naive(A, B, N, N, N);
    auto C_neon = matmul_neon(A, B, N, N, N);

    float tol = 1e-4f * static_cast<float>(N);
    expect_matrices_close(C_naive, C_neon, N, N, tol, 1e-4f);
}

TEST_P(MatMulCrossValidation, AutoMatchesNaive) {
    const size_t N = GetParam();
    auto A = random_matrix(N, N, 500);
    auto B = random_matrix(N, N, 600);

    auto C_naive = matmul_naive(A, B, N, N, N);
    auto C_auto = matmul(A, B, N, N, N, KernelType::Auto);

    float tol = 1e-4f * static_cast<float>(N);
    expect_matrices_close(C_naive, C_auto, N, N, tol, 1e-4f);
}

// Test sizes that stress remainder handling:
// - 4: exactly one 4×4 micro-kernel, no remainder
// - 7: remainder in both M and N (7 % 4 = 3)
// - 16: multiple micro-kernels, no remainder
// - 33: remainder (33 % 4 = 1)
// - 64: full tile, no remainder
// - 65: one element remainder after full tile
// - 128: multiple full tiles
// - 256: large enough to exercise KC tiling
INSTANTIATE_TEST_SUITE_P(
    Sizes,
    MatMulCrossValidation,
    ::testing::Values(4, 7, 16, 33, 64, 65, 128, 256)
);

// ─── Non-Square Cross-Validation ─────────────────────────────────────────────

struct NonSquareParams {
    size_t M, K, N;
};

class MatMulNonSquare : public ::testing::TestWithParam<NonSquareParams> {};

TEST_P(MatMulNonSquare, AllKernelsMatch) {
    auto [M, K, N] = GetParam();
    auto A = random_matrix(M, K, 700);
    auto B = random_matrix(K, N, 800);

    auto C_naive = matmul_naive(A, B, M, K, N);
    auto C_tiled = matmul_tiled(A, B, M, K, N);
    auto C_neon = matmul_neon(A, B, M, K, N);

    float tol = 1e-4f * static_cast<float>(std::max({M, K, N}));
    expect_matrices_close(C_naive, C_tiled, M, N, tol, 1e-4f);
    expect_matrices_close(C_naive, C_neon, M, N, tol, 1e-4f);
}

INSTANTIATE_TEST_SUITE_P(
    MLShapes,
    MatMulNonSquare,
    ::testing::Values(
        NonSquareParams{1, 64, 64},      // Single vector × matrix (embedding lookup)
        NonSquareParams{32, 64, 128},    // Small batch attention
        NonSquareParams{128, 768, 3072}, // GPT-2 FFN expand
        NonSquareParams{128, 3072, 768}, // GPT-2 FFN contract
        NonSquareParams{5, 5, 5},        // Tiny (tests naive path)
        NonSquareParams{3, 100, 7},      // Skinny tall × skinny wide
        NonSquareParams{100, 3, 100}     // Wide × tall
    )
);

// ─── Edge Cases ──────────────────────────────────────────────────────────────

TEST(MatMulKernels, SingleElement) {
    std::vector<float> A = {3.0f};
    std::vector<float> B = {7.0f};
    auto C = matmul(A, B, 1, 1, 1);
    EXPECT_NEAR(C[0], 21.0f, 1e-6f);
}

TEST(MatMulKernels, RowTimesColumn) {
    // [1 2 3] × [4] = [1*4 + 2*5 + 3*6] = [32]
    //            [5]
    //            [6]
    std::vector<float> A = {1, 2, 3};
    std::vector<float> B = {4, 5, 6};
    auto C = matmul(A, B, 1, 3, 1);
    EXPECT_NEAR(C[0], 32.0f, 1e-6f);
}

TEST(MatMulKernels, ZeroMatrix) {
    const size_t N = 32;
    std::vector<float> A(N * N, 0.0f);
    auto B = random_matrix(N, N, 999);
    auto C = matmul(A, B, N, N, N);
    for (size_t i = 0; i < N * N; ++i) {
        EXPECT_EQ(C[i], 0.0f);
    }
}

// ─── Dispatch Heuristic ──────────────────────────────────────────────────────

TEST(MatMulDispatch, TinyUsesNaive) {
    EXPECT_EQ(select_kernel(4, 4, 4), KernelType::Naive);
    EXPECT_EQ(select_kernel(8, 8, 8), KernelType::Naive);
    EXPECT_EQ(select_kernel(15, 15, 15), KernelType::Naive);
}

TEST(MatMulDispatch, MediumUsesTiled) {
    EXPECT_EQ(select_kernel(32, 32, 32), KernelType::Tiled);
    EXPECT_EQ(select_kernel(48, 48, 48), KernelType::Tiled);
}

TEST(MatMulDispatch, LargeUsesNeon) {
    EXPECT_EQ(select_kernel(64, 64, 64), KernelType::Neon);
    EXPECT_EQ(select_kernel(256, 256, 256), KernelType::Neon);
    EXPECT_EQ(select_kernel(1024, 1024, 1024), KernelType::Neon);
}

TEST(MatMulDispatch, AsymmetricLargeUsesNeon) {
    // If max_dim >= 256, should use NEON even if min_dim is small
    EXPECT_EQ(select_kernel(32, 32, 256), KernelType::Neon);
    EXPECT_EQ(select_kernel(16, 768, 3072), KernelType::Neon);
}

// ─── GFLOPS Calculation ──────────────────────────────────────────────────────

TEST(MatMulDispatch, GflopsCalculation) {
    // 512×512×512 in 0.5 seconds = 2*512³ / 0.5e9 = 0.537 GFLOPS
    double gflops = compute_gflops(512, 512, 512, 0.5);
    EXPECT_NEAR(gflops, 0.5369, 0.001);
}
