#pragma once

/// @file parallel_for.h
/// @brief Parallel loop utilities for intra-op parallelism.
///
/// Intra-op parallelism splits a single operation across multiple threads:
///   - MatMul: each thread computes a block of output rows
///   - ReLU: each thread processes a chunk of elements
///   - Softmax: each thread handles a subset of rows
///
/// parallel_for(pool, 0, M, [&](size_t start, size_t end) {
///     for (size_t i = start; i < end; ++i) {
///         compute_row(i);
///     }
/// });
///
/// This splits [0, M) into chunks, one per thread, and waits for all to finish.
///
/// Chunk sizing strategy:
///   - Too few chunks (< num_threads): some cores idle
///   - Too many chunks (1 per element): dispatch overhead dominates
///   - Sweet spot: 1-4 chunks per thread (amortize overhead, good load balance)
///
/// For very small ranges (< 1024 elements), we skip parallelism entirely
/// because the dispatch overhead (~500ns) exceeds the computation time.
///
/// Real-world parallels:
///   - OpenMP: #pragma omp parallel for
///   - oneDNN: parallel_nd() for multi-dimensional parallel loops
///   - Eigen: parallelFor() in ThreadPoolDevice
///   - TBB: tbb::parallel_for(range, body)

#include <inferx/parallel/thread_pool.h>

#include <cstddef>
#include <algorithm>
#include <vector>
#include <future>
#include <functional>

namespace inferx::parallel {

/// Minimum range size to justify parallelism.
/// Below this, sequential execution is faster due to dispatch overhead.
inline constexpr size_t kParallelThreshold = 1024;

/// parallel_for: split [begin, end) into chunks and execute in parallel.
///
/// @param pool     Thread pool to dispatch work to
/// @param begin    Start of range (inclusive)
/// @param end      End of range (exclusive)
/// @param body     Callable(size_t chunk_start, size_t chunk_end)
/// @param grain    Minimum chunk size (optional, 0 = auto)
///
/// The body receives [chunk_start, chunk_end) — a contiguous subrange to process.
/// All chunks complete before parallel_for returns (synchronous).
template <typename Body>
void parallel_for(ThreadPool& pool, size_t begin, size_t end, Body&& body,
                  size_t grain = 0) {
    if (begin >= end) return;

    size_t total = end - begin;

    // Skip parallelism for small ranges
    if (total < kParallelThreshold || pool.num_threads() <= 1) {
        body(begin, end);
        return;
    }

    // Compute chunk size
    size_t num_chunks = pool.num_threads();
    if (grain > 0) {
        num_chunks = std::min(num_chunks, (total + grain - 1) / grain);
    }
    num_chunks = std::max(num_chunks, size_t(1));
    size_t chunk_size = (total + num_chunks - 1) / num_chunks;

    // Submit chunks to pool
    std::vector<std::future<void>> futures;
    futures.reserve(num_chunks);

    for (size_t i = 0; i < num_chunks; ++i) {
        size_t chunk_begin = begin + i * chunk_size;
        size_t chunk_end = std::min(chunk_begin + chunk_size, end);
        if (chunk_begin >= end) break;

        futures.push_back(pool.submit([&body, chunk_begin, chunk_end] {
            body(chunk_begin, chunk_end);
        }));
    }

    // Wait for all chunks to complete
    for (auto& f : futures) {
        f.get();
    }
}

/// parallel_for with index-based body: body(size_t index) called for each element.
/// Internally groups into chunks for efficiency.
template <typename Body>
void parallel_for_each(ThreadPool& pool, size_t begin, size_t end, Body&& body) {
    parallel_for(pool, begin, end, [&body](size_t start, size_t stop) {
        for (size_t i = start; i < stop; ++i) {
            body(i);
        }
    });
}

/// Parallel reduction: compute an aggregate over [begin, end) in parallel.
///
/// @param pool     Thread pool
/// @param begin    Start of range
/// @param end      End of range
/// @param identity Initial value for reduction (e.g., 0 for sum)
/// @param map      Callable(size_t index) → T (maps index to value)
/// @param reduce   Callable(T, T) → T (combines two values)
///
/// Example: parallel sum
///   float sum = parallel_reduce(pool, 0, N, 0.0f,
///       [&](size_t i) { return data[i]; },
///       [](float a, float b) { return a + b; });
template <typename T, typename MapFn, typename ReduceFn>
T parallel_reduce(ThreadPool& pool, size_t begin, size_t end,
                  T identity, MapFn&& map_fn, ReduceFn&& reduce_fn) {
    if (begin >= end) return identity;

    size_t total = end - begin;

    // Sequential for small ranges
    if (total < kParallelThreshold || pool.num_threads() <= 1) {
        T result = identity;
        for (size_t i = begin; i < end; ++i) {
            result = reduce_fn(result, map_fn(i));
        }
        return result;
    }

    size_t num_chunks = pool.num_threads();
    size_t chunk_size = (total + num_chunks - 1) / num_chunks;

    // Each chunk produces a partial result
    std::vector<std::future<T>> futures;
    futures.reserve(num_chunks);

    for (size_t c = 0; c < num_chunks; ++c) {
        size_t chunk_begin = begin + c * chunk_size;
        size_t chunk_end = std::min(chunk_begin + chunk_size, end);
        if (chunk_begin >= end) break;

        futures.push_back(pool.submit(
            [&map_fn, &reduce_fn, identity, chunk_begin, chunk_end] {
                T partial = identity;
                for (size_t i = chunk_begin; i < chunk_end; ++i) {
                    partial = reduce_fn(partial, map_fn(i));
                }
                return partial;
            }
        ));
    }

    // Combine partial results
    T result = identity;
    for (auto& f : futures) {
        result = reduce_fn(result, f.get());
    }
    return result;
}

} // namespace inferx::parallel
