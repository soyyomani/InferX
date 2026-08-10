#pragma once

/// @file matmul_dispatch.h
/// @brief Runtime kernel dispatch for matrix multiplication.
///
/// Why dispatch? Different matrix sizes benefit from different strategies:
///   - Tiny (M,N,K < 16):    Naive is fine — tiling overhead hurts more than helps
///   - Medium (16-64):       Tiled wins — cache blocking pays off
///   - Large (>= 64):        NEON-tiled — SIMD + cache blocking = maximum throughput
///
/// This is how real inference runtimes (TensorRT, oneDNN, XNNPACK) work:
/// they maintain multiple kernel implementations and select the best one at runtime
/// based on problem dimensions, alignment, and hardware capabilities.
///
/// Usage:
///   inferx::kernels::matmul(A, B, C, M, K, N);  // auto-selects best kernel
///   inferx::kernels::matmul(A, B, C, M, K, N, KernelType::Neon);  // force specific

#include <inferx/kernels/matmul_naive.h>
#include <inferx/kernels/matmul_tiled.h>
#include <inferx/kernels/matmul_neon.h>

#include <cstddef>
#include <vector>
#include <string>

namespace inferx::kernels {

/// Available kernel implementations
enum class KernelType {
    Naive,      ///< Triple-loop baseline (~0.5 GFLOPS)
    Tiled,      ///< Cache-blocked (~3-8 GFLOPS)
    Neon,       ///< NEON SIMD + tiled (~10-20 GFLOPS)
    Auto        ///< Runtime selection based on dimensions
};

/// Convert KernelType to string (for benchmarking output / logging)
inline std::string kernel_type_name(KernelType type) {
    switch (type) {
        case KernelType::Naive: return "Naive";
        case KernelType::Tiled: return "Tiled";
        case KernelType::Neon:  return "NEON";
        case KernelType::Auto:  return "Auto";
    }
    return "Unknown";
}

/// Heuristic: select the best kernel for given dimensions.
///
/// Decision logic:
///   1. If any dimension < 16: tiling/SIMD overhead isn't worth it → Naive
///   2. If all dimensions >= 64 and NEON available: full SIMD pipeline → Neon
///   3. Otherwise: cache blocking helps but SIMD micro-kernel may not fill → Tiled
///
/// In production runtimes, this would also consider:
///   - Memory alignment (16-byte for NEON loads)
///   - Whether matrices are transposed
///   - Available cores / current thread count
///   - L1/L2/L3 cache sizes (queried at runtime or compile-time)
inline KernelType select_kernel(size_t M, size_t K, size_t N) {
    const size_t min_dim = std::min({M, K, N});
    const size_t max_dim = std::max({M, K, N});

    // Tiny matrices: overhead of tiling hurts
    if (min_dim < 16) {
        return KernelType::Naive;
    }

    // Large matrices: NEON shines
    if (min_dim >= 64 || max_dim >= 256) {
        return KernelType::Neon;
    }

    // Medium: tiling helps, SIMD micro-kernel may not fully saturate
    return KernelType::Tiled;
}

/// Dispatched matrix multiply: C = A × B
/// Automatically selects the best kernel, or use a specific one.
///
/// @param A     Row-major [M × K]
/// @param B     Row-major [K × N]
/// @param C     Row-major [M × N] (output, must be pre-allocated)
/// @param M     Rows of A
/// @param K     Cols of A / rows of B
/// @param N     Cols of B
/// @param type  Kernel to use (default: Auto)
inline void matmul(const float* __restrict__ A,
                   const float* __restrict__ B,
                   float* __restrict__ C,
                   size_t M, size_t K, size_t N,
                   KernelType type = KernelType::Auto) {
    if (type == KernelType::Auto) {
        type = select_kernel(M, K, N);
    }

    switch (type) {
        case KernelType::Naive:
            matmul_naive(A, B, C, M, K, N);
            break;
        case KernelType::Tiled:
            matmul_tiled(A, B, C, M, K, N);
            break;
        case KernelType::Neon:
            matmul_neon(A, B, C, M, K, N);
            break;
        case KernelType::Auto:
            // Unreachable — resolved above
            matmul_neon(A, B, C, M, K, N);
            break;
    }
}

/// Convenience wrapper returning a vector
inline std::vector<float> matmul(const std::vector<float>& A,
                                  const std::vector<float>& B,
                                  size_t M, size_t K, size_t N,
                                  KernelType type = KernelType::Auto) {
    std::vector<float> C(M * N);
    matmul(A.data(), B.data(), C.data(), M, K, N, type);
    return C;
}

/// Compute GFLOPS for a matmul of given dimensions and elapsed time.
/// MatMul FLOPs = 2 * M * N * K (one multiply + one add per element).
inline double compute_gflops(size_t M, size_t K, size_t N, double seconds) {
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return flops / (seconds * 1e9);
}

} // namespace inferx::kernels
