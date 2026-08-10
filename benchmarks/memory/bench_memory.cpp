/// @file bench_memory.cpp
/// @brief Allocation throughput benchmarks comparing Arena vs BufferPool vs malloc.
///
/// Simulates realistic AI inference allocation patterns:
///   - Single inference pass: allocate N tensors, use them, free all
///   - Repeated inference loop: same shapes every iteration (pool shines)
///   - Variable sizes: different batch sizes / sequence lengths
///
/// What we're measuring:
///   - Allocation throughput (allocations/second)
///   - Per-allocation latency (ns/op)
///   - Total time for a full inference memory pattern
///
/// Expected results:
///   - Arena: ~2-5 ns per allocation (pointer bump)
///   - BufferPool (warm): ~20-50 ns per allocation (hash lookup + vector pop)
///   - malloc/free: ~50-200 ns per allocation (system allocator overhead)

#include <benchmark/benchmark.h>
#include <inferx/memory/arena.h>
#include <inferx/memory/buffer_pool.h>
#include <inferx/memory/aligned_allocator.h>

#include <cstdlib>
#include <vector>

using namespace inferx::memory;

// ═══════════════════════════════════════════════════════════════════════════════
// Baseline: System malloc/free
// ═══════════════════════════════════════════════════════════════════════════════

/// Benchmark: allocate + free a single buffer using malloc
static void BM_Malloc_Single(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        void* ptr = std::malloc(size);
        benchmark::DoNotOptimize(ptr);
        std::free(ptr);
    }
    state.counters["ops/sec"] = benchmark::Counter(
        1, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_Malloc_Single)
    ->Arg(256)->Arg(1024)->Arg(4096)->Arg(65536)->Arg(1048576)
    ->Unit(benchmark::kNanosecond);

// ═══════════════════════════════════════════════════════════════════════════════
// Arena Allocator
// ═══════════════════════════════════════════════════════════════════════════════

/// Benchmark: allocate from arena (no free needed — just reset)
static void BM_Arena_Single(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    Arena arena(64 * 1024 * 1024); // 64 MB — large enough for all iterations

    for (auto _ : state) {
        void* ptr = arena.alloc_raw(size);
        benchmark::DoNotOptimize(ptr);

        // Reset periodically to avoid running out
        if (arena.remaining() < size + 64) {
            arena.reset();
        }
    }
    state.counters["ops/sec"] = benchmark::Counter(
        1, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_Arena_Single)
    ->Arg(256)->Arg(1024)->Arg(4096)->Arg(65536)->Arg(1048576)
    ->Unit(benchmark::kNanosecond);

/// Benchmark: full inference pattern — allocate N tensors, then reset
/// This is the most realistic benchmark: models the per-request memory lifecycle.
static void BM_Arena_InferencePass(benchmark::State& state) {
    const int num_tensors = static_cast<int>(state.range(0));
    Arena arena(64 * 1024 * 1024);

    // Typical tensor sizes for a transformer layer (in bytes)
    // Q, K, V: batch=32 × seq=128 × dim=768 × 4 bytes = 12.6 MB each
    // Scores: batch=32 × heads=12 × seq=128 × seq=128 × 4 = 25.2 MB
    // We scale down for benchmark practicality
    const size_t tensor_size = 32 * 128 * sizeof(float); // ~16 KB each

    for (auto _ : state) {
        for (int i = 0; i < num_tensors; ++i) {
            void* ptr = arena.alloc_raw(tensor_size);
            benchmark::DoNotOptimize(ptr);
        }
        arena.reset(); // One reset frees everything — O(1)
    }

    state.counters["allocs/iter"] = num_tensors;
    state.counters["bytes/iter"] = num_tensors * tensor_size;
}

BENCHMARK(BM_Arena_InferencePass)
    ->Arg(10)->Arg(50)->Arg(100)->Arg(500)
    ->Unit(benchmark::kMicrosecond);

/// Comparison: same pattern with malloc/free
static void BM_Malloc_InferencePass(benchmark::State& state) {
    const int num_tensors = static_cast<int>(state.range(0));
    const size_t tensor_size = 32 * 128 * sizeof(float);
    std::vector<void*> ptrs(num_tensors);

    for (auto _ : state) {
        for (int i = 0; i < num_tensors; ++i) {
            ptrs[i] = std::malloc(tensor_size);
            benchmark::DoNotOptimize(ptrs[i]);
        }
        for (int i = 0; i < num_tensors; ++i) {
            std::free(ptrs[i]);
        }
    }

    state.counters["allocs/iter"] = num_tensors;
    state.counters["bytes/iter"] = num_tensors * tensor_size;
}

BENCHMARK(BM_Malloc_InferencePass)
    ->Arg(10)->Arg(50)->Arg(100)->Arg(500)
    ->Unit(benchmark::kMicrosecond);

// ═══════════════════════════════════════════════════════════════════════════════
// Buffer Pool
// ═══════════════════════════════════════════════════════════════════════════════

/// Benchmark: BufferPool acquire/release (cold — first iteration, all misses)
static void BM_Pool_Cold(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    BufferPool pool;

    for (auto _ : state) {
        auto buf = pool.acquire(size);
        benchmark::DoNotOptimize(buf.data);
        pool.release(std::move(buf));
    }
    state.counters["ops/sec"] = benchmark::Counter(
        1, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["hit_rate"] = pool.stats().hit_rate();
}

BENCHMARK(BM_Pool_Cold)
    ->Arg(256)->Arg(1024)->Arg(4096)->Arg(65536)->Arg(1048576)
    ->Unit(benchmark::kNanosecond);

/// Benchmark: BufferPool in steady-state (warm — reuse from pool)
/// After the first acquire, subsequent ones are cache hits.
static void BM_Pool_Warm(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    BufferPool pool;

    // Warm up: one acquire + release to populate the pool
    auto warmup = pool.acquire(size);
    pool.release(std::move(warmup));

    for (auto _ : state) {
        auto buf = pool.acquire(size);
        benchmark::DoNotOptimize(buf.data);
        pool.release(std::move(buf));
    }
    state.counters["ops/sec"] = benchmark::Counter(
        1, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["hit_rate"] = pool.stats().hit_rate();
}

BENCHMARK(BM_Pool_Warm)
    ->Arg(256)->Arg(1024)->Arg(4096)->Arg(65536)->Arg(1048576)
    ->Unit(benchmark::kNanosecond);

/// Benchmark: Simulated inference loop with BufferPool
/// Same sizes every iteration — shows steady-state performance
static void BM_Pool_InferenceLoop(benchmark::State& state) {
    const int num_tensors = static_cast<int>(state.range(0));
    BufferPool pool;
    const size_t tensor_size = 32 * 128 * sizeof(float);

    std::vector<PooledBuffer> active;
    active.reserve(num_tensors);

    for (auto _ : state) {
        // Acquire all tensors
        for (int i = 0; i < num_tensors; ++i) {
            active.push_back(pool.acquire(tensor_size));
        }
        // Release all tensors (back to pool for next iteration)
        for (auto& buf : active) {
            pool.release(std::move(buf));
        }
        active.clear();
    }

    state.counters["allocs/iter"] = num_tensors;
    state.counters["hit_rate"] = pool.stats().hit_rate();
}

BENCHMARK(BM_Pool_InferenceLoop)
    ->Arg(10)->Arg(50)->Arg(100)->Arg(500)
    ->Unit(benchmark::kMicrosecond);

// ═══════════════════════════════════════════════════════════════════════════════
// Aligned Allocation
// ═══════════════════════════════════════════════════════════════════════════════

/// Benchmark: aligned_malloc vs regular malloc overhead
static void BM_AlignedMalloc(benchmark::State& state) {
    const size_t size = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        void* ptr = aligned_malloc(size, 64);
        benchmark::DoNotOptimize(ptr);
        aligned_free(ptr);
    }
    state.counters["ops/sec"] = benchmark::Counter(
        1, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_AlignedMalloc)
    ->Arg(256)->Arg(1024)->Arg(4096)->Arg(65536)->Arg(1048576)
    ->Unit(benchmark::kNanosecond);

// ═══════════════════════════════════════════════════════════════════════════════
// Head-to-Head: Full Transformer Layer Allocation Pattern
// ═══════════════════════════════════════════════════════════════════════════════

/// Simulates memory allocation for one transformer layer:
///   Q, K, V projections (3 × [batch×seq×dim])
///   Attention scores ([batch×heads×seq×seq])
///   Softmax output (same as scores)
///   Attention output ([batch×seq×dim])
///   FFN expand ([batch×seq×4×dim])
///   FFN output ([batch×seq×dim])
/// Total: 8 allocations of varying sizes

static void BM_TransformerLayer_Arena(benchmark::State& state) {
    Arena arena(32 * 1024 * 1024); // 32 MB

    // Sizes for batch=8, seq=128, dim=256, heads=4
    constexpr size_t qkv_size = 8 * 128 * 256 * sizeof(float);     // 1 MB each
    constexpr size_t scores_size = 8 * 4 * 128 * 128 * sizeof(float); // 2 MB
    constexpr size_t ffn_size = 8 * 128 * 1024 * sizeof(float);    // 4 MB

    for (auto _ : state) {
        auto* Q = arena.alloc<float>(qkv_size / sizeof(float));
        auto* K = arena.alloc<float>(qkv_size / sizeof(float));
        auto* V = arena.alloc<float>(qkv_size / sizeof(float));
        auto* scores = arena.alloc<float>(scores_size / sizeof(float));
        auto* softmax_out = arena.alloc<float>(scores_size / sizeof(float));
        auto* attn_out = arena.alloc<float>(qkv_size / sizeof(float));
        auto* ffn_hidden = arena.alloc<float>(ffn_size / sizeof(float));
        auto* ffn_out = arena.alloc<float>(qkv_size / sizeof(float));

        benchmark::DoNotOptimize(Q);
        benchmark::DoNotOptimize(K);
        benchmark::DoNotOptimize(V);
        benchmark::DoNotOptimize(scores);
        benchmark::DoNotOptimize(softmax_out);
        benchmark::DoNotOptimize(attn_out);
        benchmark::DoNotOptimize(ffn_hidden);
        benchmark::DoNotOptimize(ffn_out);

        arena.reset();
    }
}

BENCHMARK(BM_TransformerLayer_Arena)->Unit(benchmark::kNanosecond);

static void BM_TransformerLayer_Malloc(benchmark::State& state) {
    constexpr size_t qkv_size = 8 * 128 * 256 * sizeof(float);
    constexpr size_t scores_size = 8 * 4 * 128 * 128 * sizeof(float);
    constexpr size_t ffn_size = 8 * 128 * 1024 * sizeof(float);

    for (auto _ : state) {
        auto* Q = static_cast<float*>(std::malloc(qkv_size));
        auto* K = static_cast<float*>(std::malloc(qkv_size));
        auto* V = static_cast<float*>(std::malloc(qkv_size));
        auto* scores = static_cast<float*>(std::malloc(scores_size));
        auto* softmax_out = static_cast<float*>(std::malloc(scores_size));
        auto* attn_out = static_cast<float*>(std::malloc(qkv_size));
        auto* ffn_hidden = static_cast<float*>(std::malloc(ffn_size));
        auto* ffn_out = static_cast<float*>(std::malloc(qkv_size));

        benchmark::DoNotOptimize(Q);
        benchmark::DoNotOptimize(K);
        benchmark::DoNotOptimize(V);
        benchmark::DoNotOptimize(scores);
        benchmark::DoNotOptimize(softmax_out);
        benchmark::DoNotOptimize(attn_out);
        benchmark::DoNotOptimize(ffn_hidden);
        benchmark::DoNotOptimize(ffn_out);

        std::free(Q);
        std::free(K);
        std::free(V);
        std::free(scores);
        std::free(softmax_out);
        std::free(attn_out);
        std::free(ffn_hidden);
        std::free(ffn_out);
    }
}

BENCHMARK(BM_TransformerLayer_Malloc)->Unit(benchmark::kNanosecond);

static void BM_TransformerLayer_Pool(benchmark::State& state) {
    BufferPool pool;
    constexpr size_t qkv_size = 8 * 128 * 256 * sizeof(float);
    constexpr size_t scores_size = 8 * 4 * 128 * 128 * sizeof(float);
    constexpr size_t ffn_size = 8 * 128 * 1024 * sizeof(float);

    for (auto _ : state) {
        auto Q = pool.acquire(qkv_size);
        auto K = pool.acquire(qkv_size);
        auto V = pool.acquire(qkv_size);
        auto scores = pool.acquire(scores_size);
        auto softmax_out = pool.acquire(scores_size);
        auto attn_out = pool.acquire(qkv_size);
        auto ffn_hidden = pool.acquire(ffn_size);
        auto ffn_out = pool.acquire(qkv_size);

        benchmark::DoNotOptimize(Q.data);
        benchmark::DoNotOptimize(K.data);
        benchmark::DoNotOptimize(V.data);
        benchmark::DoNotOptimize(scores.data);
        benchmark::DoNotOptimize(softmax_out.data);
        benchmark::DoNotOptimize(attn_out.data);
        benchmark::DoNotOptimize(ffn_hidden.data);
        benchmark::DoNotOptimize(ffn_out.data);

        pool.release(std::move(Q));
        pool.release(std::move(K));
        pool.release(std::move(V));
        pool.release(std::move(scores));
        pool.release(std::move(softmax_out));
        pool.release(std::move(attn_out));
        pool.release(std::move(ffn_hidden));
        pool.release(std::move(ffn_out));
    }
    state.counters["hit_rate"] = pool.stats().hit_rate();
}

BENCHMARK(BM_TransformerLayer_Pool)->Unit(benchmark::kNanosecond);
