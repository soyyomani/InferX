#pragma once

/// @file matmul_neon.h
/// @brief ARM NEON SIMD-accelerated tiled matrix multiplication.
///
/// This is the high-performance kernel that combines two optimizations:
///   1. Cache tiling (from matmul_tiled.h) — keeps working set in L1
///   2. NEON SIMD intrinsics — processes 4 floats per instruction
///
/// ARM NEON on Apple M1:
///   - 128-bit vector registers (32 registers: v0-v31)
///   - float32x4_t = 4 floats per register
///   - vfmaq_f32: fused multiply-accumulate (2 FLOPs per element, 8 per instruction)
///   - Peak: ~2.4 TFLOPS (with AMX), ~50 GFLOPS (NEON only, single core)
///
/// Strategy: 4×4 micro-kernel
///   - Load 4 elements of C into 4 NEON registers (one row of 4×4 tile)
///   - For each k: broadcast A[i][k], load B[k][j:j+4], fused multiply-add
///   - This gives 4×4 = 16 FMAs per inner iteration = 32 FLOPs
///   - Register usage: 4 accumulators + 1 A broadcast + 1 B load = 6 registers
///
/// Expected performance: 10-20 GFLOPS on M1 (single core, NEON only)
/// That's ~15-40× faster than naive.

#include <cstddef>
#include <cstring>
#include <algorithm>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define INFERX_HAS_NEON 1
#else
#define INFERX_HAS_NEON 0
#endif

namespace inferx::kernels {

#if INFERX_HAS_NEON

/// 4×4 micro-kernel: computes a 4×4 block of C using NEON.
/// Accumulates: C[i:i+4][j:j+4] += A[i:i+4][k] * B[k][j:j+4] for all k in range.
///
/// This is the innermost compute-bound loop — everything else exists to
/// feed data to this kernel efficiently.
inline void microkernel_4x4_neon(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  size_t K, size_t N,
                                  size_t k_start, size_t k_end) {
    // 4 accumulator registers — one per row of the 4×4 output tile
    // Each holds [C[row][j], C[row][j+1], C[row][j+2], C[row][j+3]]
    float32x4_t c0 = vld1q_f32(C + 0 * N);
    float32x4_t c1 = vld1q_f32(C + 1 * N);
    float32x4_t c2 = vld1q_f32(C + 2 * N);
    float32x4_t c3 = vld1q_f32(C + 3 * N);

    for (size_t k = k_start; k < k_end; ++k) {
        // Load one column of B: B[k][j:j+4] — 4 consecutive floats
        float32x4_t b = vld1q_f32(B + k * N);

        // Broadcast A[row][k] and fused multiply-accumulate
        // vfmaq_f32(acc, a, b) = acc + a * b (element-wise, fused)
        c0 = vfmaq_f32(c0, vdupq_n_f32(A[0 * K + k]), b);
        c1 = vfmaq_f32(c1, vdupq_n_f32(A[1 * K + k]), b);
        c2 = vfmaq_f32(c2, vdupq_n_f32(A[2 * K + k]), b);
        c3 = vfmaq_f32(c3, vdupq_n_f32(A[3 * K + k]), b);
    }

    // Store results back to C
    vst1q_f32(C + 0 * N, c0);
    vst1q_f32(C + 1 * N, c1);
    vst1q_f32(C + 2 * N, c2);
    vst1q_f32(C + 3 * N, c3);
}

/// Tiled + NEON matrix multiply: C = A × B
/// @param A     Row-major [M × K]
/// @param B     Row-major [K × N]
/// @param C     Row-major [M × N] (output, must be pre-allocated)
/// @param M     Rows of A
/// @param K     Cols of A / rows of B
/// @param N     Cols of B
///
/// Algorithm:
///   1. Outer tiling loop partitions into cache-friendly blocks
///   2. Within each tile, dispatch 4×4 micro-kernels for the aligned portion
///   3. Handle remainder edges with scalar code
///
/// Performance:
///   - 4×4 micro-kernel: 32 FLOPs per k-iteration, high register utilization
///   - Tiling ensures A/B tiles stay in L1 during micro-kernel execution
///   - Expected: 10-20 GFLOPS on Apple M1 (single core)
inline void matmul_neon(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        size_t M, size_t K, size_t N) {
    // Zero output
    std::memset(C, 0, M * N * sizeof(float));

    // Tile sizes tuned for M1:
    // - MC (M-tile): 64 rows of A at a time
    // - KC (K-tile): 256 depth — fits A-panel + B-panel in L2
    // - NC (N-tile): 64 cols of B at a time
    constexpr size_t MC = 64;   // M-direction cache block
    constexpr size_t KC = 256;  // K-direction cache block
    constexpr size_t NC = 64;   // N-direction cache block

    // Micro-kernel dimensions
    constexpr size_t MR = 4;    // Rows per micro-kernel
    constexpr size_t NR = 4;    // Cols per micro-kernel (NEON width)

    for (size_t ii = 0; ii < M; ii += MC) {
        const size_t i_end = std::min(ii + MC, M);

        for (size_t kk = 0; kk < K; kk += KC) {
            const size_t k_end = std::min(kk + KC, K);

            for (size_t jj = 0; jj < N; jj += NC) {
                const size_t j_end = std::min(jj + NC, N);

                // Process 4×4 micro-kernels within this tile
                size_t i = ii;
                for (; i + MR <= i_end; i += MR) {
                    size_t j = jj;
                    for (; j + NR <= j_end; j += NR) {
                        // Dispatch the 4×4 NEON micro-kernel
                        microkernel_4x4_neon(
                            A + i * K,          // A pointer (row i, col 0)
                            B + j,              // B pointer (row 0, col j) — add k*N inside
                            C + i * N + j,      // C pointer (row i, col j)
                            K, N,
                            kk, k_end
                        );
                    }

                    // Remainder columns (< 4 wide): scalar fallback
                    for (; j < j_end; ++j) {
                        for (size_t k = kk; k < k_end; ++k) {
                            C[(i + 0) * N + j] += A[(i + 0) * K + k] * B[k * N + j];
                            C[(i + 1) * N + j] += A[(i + 1) * K + k] * B[k * N + j];
                            C[(i + 2) * N + j] += A[(i + 2) * K + k] * B[k * N + j];
                            C[(i + 3) * N + j] += A[(i + 3) * K + k] * B[k * N + j];
                        }
                    }
                }

                // Remainder rows (< 4 tall): scalar fallback
                for (; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        const float a_ik = A[i * K + k];
                        for (size_t j = jj; j < j_end; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

#else // !INFERX_HAS_NEON

/// Fallback: on non-ARM platforms, use the tiled implementation
/// This ensures the code compiles everywhere (x86, CI, etc.)
inline void matmul_neon(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        size_t M, size_t K, size_t N) {
    // Zero output
    std::memset(C, 0, M * N * sizeof(float));

    constexpr size_t tile = 64;
    for (size_t ii = 0; ii < M; ii += tile) {
        const size_t i_end = std::min(ii + tile, M);
        for (size_t kk = 0; kk < K; kk += tile) {
            const size_t k_end = std::min(kk + tile, K);
            for (size_t jj = 0; jj < N; jj += tile) {
                const size_t j_end = std::min(jj + tile, N);
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        const float a_ik = A[i * K + k];
                        for (size_t j = jj; j < j_end; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

#endif // INFERX_HAS_NEON

/// Convenience wrapper returning a vector
inline std::vector<float> matmul_neon(const std::vector<float>& A,
                                      const std::vector<float>& B,
                                      size_t M, size_t K, size_t N) {
    std::vector<float> C(M * N);
    matmul_neon(A.data(), B.data(), C.data(), M, K, N);
    return C;
}

} // namespace inferx::kernels
