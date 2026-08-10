#pragma once

/// @file buffer_pool.h
/// @brief Size-bucketed tensor buffer reuse pool.
///
/// Why a buffer pool?
/// ──────────────────
/// In inference servers handling continuous requests, the same tensor shapes
/// appear repeatedly (same model, same batch size). Instead of allocating and
/// freeing memory for every request:
///
///   Request 1: malloc(768*4) → use → free(768*4)
///   Request 2: malloc(768*4) → use → free(768*4)  // same size!
///   Request 3: malloc(768*4) → use → free(768*4)  // again!
///
/// A buffer pool keeps freed buffers in buckets by size class, and hands them
/// back on the next allocation of the same (or similar) size:
///
///   Request 1: pool.acquire(768*4) → [cache miss: malloc] → pool.release()
///   Request 2: pool.acquire(768*4) → [cache hit: reuse!]  → pool.release()
///   Request 3: pool.acquire(768*4) → [cache hit: reuse!]  → pool.release()
///
/// After warm-up, the pool achieves ~100% reuse rate — zero allocations on
/// the hot path.
///
/// Size bucketing:
///   We round up requested sizes to the nearest power-of-2 bucket. This means
///   a 3000-byte request uses a 4096-byte buffer (33% waste), but enables reuse
///   across slightly different sizes. The trade-off (waste vs reuse) is tunable.
///
/// This pattern is used by:
///   - PyTorch's CUDACachingAllocator (GPU memory pool)
///   - TensorFlow's BFC allocator (best-fit with coalescing)
///   - oneDNN's memory pool for CPU tensors
///
/// Thread safety: NOT thread-safe. Use one pool per thread, or add external lock.

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <memory>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace inferx::memory {

/// Default alignment for pooled buffers (cache-line aligned)
inline constexpr size_t kPoolAlignment = 64;

/// Maximum number of buffers to keep per size bucket.
/// Beyond this, excess buffers are freed to the OS to prevent unbounded growth.
inline constexpr size_t kMaxBuffersPerBucket = 8;

/// A managed buffer with metadata for the pool.
struct PooledBuffer {
    void* data = nullptr;       ///< Pointer to aligned memory
    size_t capacity = 0;        ///< Actual allocated size (may be > requested)
    size_t bucket_size = 0;     ///< Size bucket this belongs to (power of 2)

    PooledBuffer() = default;
    PooledBuffer(void* d, size_t cap, size_t bucket)
        : data(d), capacity(cap), bucket_size(bucket) {}

    /// Check if this buffer can satisfy a request of `bytes`
    [[nodiscard]] bool fits(size_t bytes) const noexcept {
        return capacity >= bytes;
    }
};

class BufferPool {
public:
    explicit BufferPool(size_t max_per_bucket = kMaxBuffersPerBucket,
                        size_t alignment = kPoolAlignment)
        : max_per_bucket_(max_per_bucket), alignment_(alignment) {}

    ~BufferPool() {
        // Free all pooled buffers
        for (auto& [bucket, buffers] : free_lists_) {
            for (auto& buf : buffers) {
                std::free(buf.data);
            }
        }
    }

    // Non-copyable
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    // Movable
    BufferPool(BufferPool&&) = default;
    BufferPool& operator=(BufferPool&&) = default;

    /// Acquire a buffer of at least `bytes` size.
    ///
    /// Algorithm:
    ///   1. Round up to nearest power-of-2 bucket
    ///   2. Check free list for that bucket
    ///   3. If available: pop and return (cache hit — O(1), no syscall)
    ///   4. If empty: allocate fresh (cache miss — one malloc)
    ///
    /// Returns: PooledBuffer with data pointer and capacity.
    ///          Caller MUST call release() when done (not free!).
    [[nodiscard]] PooledBuffer acquire(size_t bytes) {
        if (bytes == 0) bytes = 1; // Avoid zero-size edge case

        size_t bucket = next_power_of_2(bytes);

        // Check free list
        auto it = free_lists_.find(bucket);
        if (it != free_lists_.end() && !it->second.empty()) {
            // Cache hit: reuse existing buffer
            PooledBuffer buf = std::move(it->second.back());
            it->second.pop_back();
            ++stats_.hits;
            ++stats_.active_buffers;
            return buf;
        }

        // Cache miss: allocate new buffer
        size_t alloc_size = align_up(bucket, alignment_);
        void* ptr = std::aligned_alloc(alignment_, alloc_size);
        if (!ptr) {
            throw std::bad_alloc();
        }

        ++stats_.misses;
        ++stats_.total_allocated;
        ++stats_.active_buffers;
        stats_.total_bytes_allocated += alloc_size;

        return PooledBuffer(ptr, alloc_size, bucket);
    }

    /// Release a buffer back to the pool for reuse.
    ///
    /// If the bucket is full (> max_per_bucket), the buffer is freed to OS
    /// to prevent unbounded memory growth. This is the "eviction" policy.
    void release(PooledBuffer buf) {
        if (!buf.data) return;

        --stats_.active_buffers;

        auto& bucket_list = free_lists_[buf.bucket_size];
        if (bucket_list.size() < max_per_bucket_) {
            // Room in pool — keep for reuse
            bucket_list.push_back(std::move(buf));
        } else {
            // Pool full — free to OS (eviction)
            std::free(buf.data);
            ++stats_.evictions;
        }
    }

    /// Typed acquire: get a buffer for `count` elements of type T
    template <typename T>
    [[nodiscard]] PooledBuffer acquire_typed(size_t count) {
        return acquire(count * sizeof(T));
    }

    /// Drain: free all cached buffers back to OS.
    /// Use during shutdown or when memory pressure is high.
    void drain() {
        for (auto& [bucket, buffers] : free_lists_) {
            for (auto& buf : buffers) {
                std::free(buf.data);
            }
            buffers.clear();
        }
    }

    /// Shrink: reduce each bucket to at most `keep` buffers.
    /// Useful for periodic memory pressure relief without full drain.
    void shrink(size_t keep = 1) {
        for (auto& [bucket, buffers] : free_lists_) {
            while (buffers.size() > keep) {
                std::free(buffers.back().data);
                buffers.pop_back();
                ++stats_.evictions;
            }
        }
    }

    // ─── Statistics ──────────────────────────────────────────────────────────

    struct Stats {
        size_t hits = 0;                   ///< Allocations served from pool
        size_t misses = 0;                 ///< Allocations requiring new malloc
        size_t evictions = 0;              ///< Buffers freed due to pool overflow
        size_t total_allocated = 0;        ///< Total buffers ever allocated
        size_t active_buffers = 0;         ///< Currently in-use (not returned)
        size_t total_bytes_allocated = 0;  ///< Total bytes from OS

        /// Hit rate: percentage of allocations served from cache (0.0 to 1.0)
        [[nodiscard]] double hit_rate() const noexcept {
            size_t total = hits + misses;
            return total > 0 ? static_cast<double>(hits) / static_cast<double>(total) : 0.0;
        }

        /// Total requests (hits + misses)
        [[nodiscard]] size_t total_requests() const noexcept { return hits + misses; }
    };

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

    /// Number of buffers currently cached (available for reuse)
    [[nodiscard]] size_t cached_count() const noexcept {
        size_t count = 0;
        for (const auto& [bucket, buffers] : free_lists_) {
            count += buffers.size();
        }
        return count;
    }

    /// Total bytes held in cache (not in use, but reserved)
    [[nodiscard]] size_t cached_bytes() const noexcept {
        size_t bytes = 0;
        for (const auto& [bucket, buffers] : free_lists_) {
            for (const auto& buf : buffers) {
                bytes += buf.capacity;
            }
        }
        return bytes;
    }

    /// Number of distinct size buckets currently in use
    [[nodiscard]] size_t bucket_count() const noexcept {
        return free_lists_.size();
    }

private:
    /// Round up to the next power of 2 (minimum 64 bytes to avoid tiny buckets)
    static size_t next_power_of_2(size_t n) noexcept {
        if (n <= 64) return 64;
        // Bit trick: fill all bits below the highest, then add 1
        --n;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }

    /// Round up to alignment boundary
    static constexpr size_t align_up(size_t value, size_t alignment) noexcept {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    size_t max_per_bucket_;
    size_t alignment_;
    Stats stats_;

    /// Free lists indexed by bucket size (power of 2)
    /// Key = bucket size, Value = vector of available buffers
    std::unordered_map<size_t, std::vector<PooledBuffer>> free_lists_;
};

} // namespace inferx::memory
