/// @file bench_matmul.cpp
/// @brief Google Benchmark suite comparing matmul kernel implementations.
///
/// Measures and reports GFLOPS for:
///   - Naive (baseline triple-loop)
///   - Tiled (cache-blocked)
///   - NEON (ARM SIMD + tiled)
///
/// Run with:
///   ./bench_matmul --benchmark_format=console
///   ./bench_matmul --benchmark_out=results.json --benchmark_out_format=json
///
/// Expected results on Apple M1 (512×512):
///   Naive:  ~0.5-1.0 GFLOPS
///   Tiled:  ~3-6 GFLOPS
///   NEON:   ~10-20 GFLOPS

#include <benchmark/benchmark.h>
#include <inferx/kernels/matmul_dispatch.h>

#include <vector>
#include <random>
#include <cstddef>

using namespace inferx::kernels;

/// Generate a random matrix [rows × cols] with values in [-1, 1]
static std::vector<float> random_matrix(size_t rows, size_t cols, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> m(rows * cols);
    for (auto& v : m) v = dist(gen);
    return m;
}

// ─── Benchmark Template ─────────────────────────────────────────────────────

/// Template benchmark for a specific kernel and matrix size.
/// Reports GFLOPS as a custom counter so it appears in benchmark output.
static void BM_MatMul(benchmark::State& state, KernelType kernel) {
    const size_t N = static_cast<size_t>(state.range(0));
    const size_t M = N, K = N; // Square matrices for simplicity

    auto A = random_matrix(M, K, 1);
    auto B = random_matrix(K, N, 2);
    std::vector<float> C(M * N);

    for (auto _ : state) {
        matmul(A.data(), B.data(), C.data(), M, K, N, kernel);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    // Report GFLOPS
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// ─── Register Benchmarks ─────────────────────────────────────────────────────

// Matrix sizes: 32, 64, 128, 256, 512, 1024
// These cover the range from "too small to benefit from optimization"
// to "large enough that cache behavior dominates."

// Naive kernel
BENCHMARK_CAPTURE(BM_MatMul, Naive, KernelType::Naive)
    ->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)
    ->Unit(benchmark::kMillisecond);

// Tiled kernel
BENCHMARK_CAPTURE(BM_MatMul, Tiled, KernelType::Tiled)
    ->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)
    ->Unit(benchmark::kMillisecond);

// NEON kernel
BENCHMARK_CAPTURE(BM_MatMul, NEON, KernelType::Neon)
    ->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)
    ->Unit(benchmark::kMillisecond);

// Auto-dispatch (shows what the heuristic selects)
BENCHMARK_CAPTURE(BM_MatMul, Auto, KernelType::Auto)
    ->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)
    ->Unit(benchmark::kMillisecond);

// ─── Non-Square Matrices (common in ML workloads) ────────────────────────────

/// Benchmark for typical ML shapes:
///   - Attention: [seq_len × head_dim] × [head_dim × seq_len]
///   - FFN expand: [batch × d_model] × [d_model × 4*d_model]
static void BM_MatMul_ML_Shapes(benchmark::State& state, KernelType kernel) {
    // M=128 (batch/seq), K=768 (d_model), N=3072 (4×d_model) — GPT-2 FFN shape
    const size_t M = static_cast<size_t>(state.range(0));
    const size_t K = static_cast<size_t>(state.range(1));
    const size_t N = static_cast<size_t>(state.range(2));

    auto A = random_matrix(M, K, 1);
    auto B = random_matrix(K, N, 2);
    std::vector<float> C(M * N);

    for (auto _ : state) {
        matmul(A.data(), B.data(), C.data(), M, K, N, kernel);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }

    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// GPT-2 style shapes
BENCHMARK_CAPTURE(BM_MatMul_ML_Shapes, NEON_Attention_QK, KernelType::Neon)
    ->Args({128, 64, 128})   // seq=128, head_dim=64, seq=128 (Q×K^T)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_MatMul_ML_Shapes, NEON_FFN_Expand, KernelType::Neon)
    ->Args({32, 768, 3072})  // batch=32, d_model=768, 4×d_model=3072
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_MatMul_ML_Shapes, NEON_FFN_Contract, KernelType::Neon)
    ->Args({32, 3072, 768})  // batch=32, 4×d_model=3072, d_model=768
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(BM_MatMul_ML_Shapes, NEON_Embedding, KernelType::Neon)
    ->Args({128, 768, 768})  // seq=128, d_model=768, d_model=768
    ->Unit(benchmark::kMillisecond);
