#pragma once

/// @file matmul_naive.h
/// @brief Naive O(N³) matrix multiplication — baseline for benchmarking.
///
/// This is the textbook triple-loop matmul. It exists purely as a performance
/// reference point: any optimized kernel should beat this by 10-50x on real
/// hardware due to poor cache utilization (strided access on B columns).
///
/// Layout: Row-major. C[i][j] += A[i][k] * B[k][j]
/// Complexity: 2*M*N*K FLOPs

#include <cstddef>
#include <cstring>
#include <vector>

namespace inferx::kernels {

/// Naive matrix multiply: C = A × B
/// @param A  Row-major [M × K]
/// @param B  Row-major [K × N]
/// @param C  Row-major [M × N] (output, must be pre-allocated)
/// @param M  Rows of A / rows of C
/// @param K  Cols of A / rows of B
/// @param N  Cols of B / cols of C
///
/// Performance characteristics:
/// - No cache blocking → every B column access strides by N floats
/// - No vectorization hints → compiler may auto-vectorize inner loop
/// - Expected: ~0.3-0.8 GFLOPS on Apple M1 for 512×512
inline void matmul_naive(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         size_t M, size_t K, size_t N) {
    // Zero output
    std::memset(C, 0, M * N * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const float a_ik = A[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

/// Convenience wrapper returning a vector
inline std::vector<float> matmul_naive(const std::vector<float>& A,
                                       const std::vector<float>& B,
                                       size_t M, size_t K, size_t N) {
    std::vector<float> C(M * N);
    matmul_naive(A.data(), B.data(), C.data(), M, K, N);
    return C;
}

} // namespace inferx::kernels
