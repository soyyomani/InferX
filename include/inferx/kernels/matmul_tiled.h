#pragma once

/// @file matmul_tiled.h
/// @brief Cache-blocked (tiled) matrix multiplication.
///
/// Key insight: The naive matmul thrashes cache because B is accessed column-wise,
/// pulling in cache lines that get evicted before reuse. Tiling partitions the
/// computation into small blocks that fit entirely in L1/L2 cache.
///
/// Apple M1 cache sizes:
///   L1 data: 128 KB (per P-core), 64-byte cache lines
///   L2:      12 MB (shared)
///
/// For a tile of size T×T floats:
///   3 tiles (A_tile, B_tile, C_tile) × T² × 4 bytes
///   T=32: 3 × 1024 × 4 = 12 KB → fits comfortably in L1
///   T=64: 3 × 4096 × 4 = 48 KB → still fits in L1
///   T=128: 3 × 16384 × 4 = 192 KB → spills to L2, still fast
///
/// Layout: Row-major. C[i][j] += A[i][k] * B[k][j]
/// Complexity: Still 2*M*N*K FLOPs, but much better cache utilization.
/// Expected: 3-8× speedup over naive due to reduced cache misses.

#include <cstddef>
#include <cstring>
#include <algorithm>
#include <vector>

namespace inferx::kernels {

/// Default tile size — tuned for Apple M1 L1 cache (128 KB).
/// 3 tiles of 64×64 floats = 48 KB, well within L1.
inline constexpr size_t kDefaultTileSize = 64;

/// Cache-blocked matrix multiply: C = A × B
/// @param A     Row-major [M × K]
/// @param B     Row-major [K × N]
/// @param C     Row-major [M × N] (output, must be pre-allocated)
/// @param M     Rows of A / rows of C
/// @param K     Cols of A / rows of B
/// @param N     Cols of B / cols of C
/// @param tile  Tile size (controls L1 working set)
///
/// Performance characteristics:
/// - Each tile computation reuses data in L1 cache
/// - Inner loop accesses are sequential (row-major friendly)
/// - ~3-8× faster than naive for matrices > 128×128
/// - Diminishing returns for very small matrices (overhead of tile logic)
inline void matmul_tiled(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         size_t M, size_t K, size_t N,
                         size_t tile = kDefaultTileSize) {
    // Zero output
    std::memset(C, 0, M * N * sizeof(float));

    // Tiled iteration: partition M, N, K into blocks of size `tile`
    // For each tile of C[ii..ii+tile][jj..jj+tile], accumulate contributions
    // from all K-tiles of A and B.
    for (size_t ii = 0; ii < M; ii += tile) {
        const size_t i_end = std::min(ii + tile, M);

        for (size_t kk = 0; kk < K; kk += tile) {
            const size_t k_end = std::min(kk + tile, K);

            for (size_t jj = 0; jj < N; jj += tile) {
                const size_t j_end = std::min(jj + tile, N);

                // Micro-kernel: multiply tile of A × tile of B → accumulate into tile of C
                // At this point, the working set is:
                //   A[ii:i_end, kk:k_end]  — up to tile×tile floats
                //   B[kk:k_end, jj:j_end]  — up to tile×tile floats
                //   C[ii:i_end, jj:j_end]  — up to tile×tile floats
                // All three fit in L1 cache simultaneously.
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

/// Convenience wrapper returning a vector
inline std::vector<float> matmul_tiled(const std::vector<float>& A,
                                       const std::vector<float>& B,
                                       size_t M, size_t K, size_t N,
                                       size_t tile = kDefaultTileSize) {
    std::vector<float> C(M * N);
    matmul_tiled(A.data(), B.data(), C.data(), M, K, N, tile);
    return C;
}

} // namespace inferx::kernels
