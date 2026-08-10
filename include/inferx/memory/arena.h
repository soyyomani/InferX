#pragma once

/// @file arena.h
/// @brief Bump (arena) allocator for AI inference scratch memory.
///
/// Why an arena allocator for inference?
/// ─────────────────────────────────────
/// During a single inference pass, the runtime allocates many temporary tensors:
///   - Attention: Q, K, V projections, scores, softmax output
///   - FFN: expanded hidden state (4× model dim), GELU intermediate
///   - Each layer produces intermediates that die before the next layer
///
/// With malloc/free:
///   - Each allocation hits the system allocator (locks, fragmentation)
///   - Free doesn't return memory to OS immediately
///   - Fragmentation grows over time in long-running servers
///
/// With an arena:
///   - Pre-allocate one large block (e.g., 64 MB)
///   - Each "allocation" = bump a pointer (1 instruction, no lock)
///   - After inference completes, reset pointer to start (instant "free all")
///   - Zero fragmentation, perfect cache locality for sequential access
///
/// This is exactly how TensorRT, CoreML, and XLA manage inference memory.
///
/// Usage:
///   Arena arena(64 * 1024 * 1024);  // 64 MB pre-allocated
///   float* Q = arena.alloc<float>(seq_len * d_model);
///   float* K = arena.alloc<float>(seq_len * d_model);
///   // ... do inference ...
///   arena.reset();  // instant: all memory "freed", ready for next request
///
/// Thread safety: NOT thread-safe by design. Each thread should own its arena.
/// (Same pattern as Google's tcmalloc thread-local caches.)

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace inferx::memory {

/// Cache line size on most modern CPUs (Apple M1, Intel, AMD)
inline constexpr size_t kCacheLineSize = 64;

/// Default arena capacity: 16 MB — enough for a single transformer layer
/// at batch=32, seq_len=512, d_model=768 (~75 MB total, but intermediates
/// are reused across layers so 16 MB per arena is sufficient with careful
/// reset points).
inline constexpr size_t kDefaultArenaCapacity = 16 * 1024 * 1024;

class Arena {
public:
    /// Construct arena with given capacity in bytes.
    /// Memory is allocated immediately (one-time cost at startup).
    explicit Arena(size_t capacity = kDefaultArenaCapacity)
        : capacity_(capacity), offset_(0) {
        if (capacity == 0) {
            throw std::invalid_argument("Arena: capacity must be > 0");
        }
        // Allocate aligned memory for the entire arena
        // aligned_alloc requires size to be a multiple of alignment
        size_t aligned_capacity = align_up(capacity, kCacheLineSize);
        base_ = static_cast<uint8_t*>(std::aligned_alloc(kCacheLineSize, aligned_capacity));
        if (!base_) {
            throw std::bad_alloc();
        }
        capacity_ = aligned_capacity;
    }

    ~Arena() {
        std::free(base_);
    }

    // Non-copyable (owns raw memory)
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Movable
    Arena(Arena&& other) noexcept
        : base_(other.base_), capacity_(other.capacity_),
          offset_(other.offset_), peak_offset_(other.peak_offset_),
          allocation_count_(other.allocation_count_) {
        other.base_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
    }

    Arena& operator=(Arena&& other) noexcept {
        if (this != &other) {
            std::free(base_);
            base_ = other.base_;
            capacity_ = other.capacity_;
            offset_ = other.offset_;
            peak_offset_ = other.peak_offset_;
            allocation_count_ = other.allocation_count_;
            other.base_ = nullptr;
            other.capacity_ = 0;
            other.offset_ = 0;
        }
        return *this;
    }

    /// Allocate `count` elements of type T, aligned to `alignment` bytes.
    /// Returns nullptr if arena is exhausted (no exception — fail-fast pattern).
    ///
    /// Cost: ~3 instructions (align + bump + bounds check). No syscall, no lock.
    template <typename T>
    [[nodiscard]] T* alloc(size_t count, size_t alignment = alignof(T)) {
        size_t bytes = count * sizeof(T);
        return static_cast<T*>(alloc_raw(bytes, alignment));
    }

    /// Raw byte allocation with specified alignment.
    /// This is the core allocation primitive.
    ///
    /// Algorithm:
    ///   1. Align current offset up to requested alignment
    ///   2. Check if aligned_offset + size fits in capacity
    ///   3. Bump offset past the allocation
    ///   4. Return pointer to aligned position
    [[nodiscard]] void* alloc_raw(size_t bytes, size_t alignment = kCacheLineSize) {
        // Align the current offset
        size_t aligned_offset = align_up(offset_, alignment);

        // Check capacity
        if (aligned_offset + bytes > capacity_) {
            return nullptr; // Out of memory — caller decides policy
        }

        // Bump the offset
        void* ptr = base_ + aligned_offset;
        offset_ = aligned_offset + bytes;

        // Track stats
        ++allocation_count_;
        if (offset_ > peak_offset_) {
            peak_offset_ = offset_;
        }

        return ptr;
    }

    /// Reset the arena — "frees" all allocations instantly.
    /// After reset, the same memory can be reused for the next inference.
    ///
    /// Cost: 1 instruction (set offset = 0). Compared to free-ing N tensors
    /// individually (N syscalls), this is effectively free.
    void reset() noexcept {
        offset_ = 0;
        // Note: peak_offset_ and allocation_count_ are NOT reset.
        // They accumulate across the arena's lifetime for profiling.
    }

    /// Hard reset — clears stats too. Use between benchmark iterations.
    void hard_reset() noexcept {
        offset_ = 0;
        peak_offset_ = 0;
        allocation_count_ = 0;
    }

    // ─── Accessors ───────────────────────────────────────────────────────────

    /// Total capacity in bytes
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }

    /// Currently used bytes (offset from base)
    [[nodiscard]] size_t used() const noexcept { return offset_; }

    /// Remaining available bytes
    [[nodiscard]] size_t remaining() const noexcept { return capacity_ - offset_; }

    /// Peak usage across all allocations (high-water mark)
    [[nodiscard]] size_t peak_usage() const noexcept { return peak_offset_; }

    /// Total number of allocations since construction (or hard_reset)
    [[nodiscard]] size_t allocation_count() const noexcept { return allocation_count_; }

    /// Utilization ratio: peak_usage / capacity (0.0 to 1.0)
    [[nodiscard]] double utilization() const noexcept {
        return capacity_ > 0 ? static_cast<double>(peak_offset_) / static_cast<double>(capacity_) : 0.0;
    }

    /// Check if a pointer was allocated from this arena
    [[nodiscard]] bool owns(const void* ptr) const noexcept {
        auto p = static_cast<const uint8_t*>(ptr);
        return p >= base_ && p < base_ + capacity_;
    }

    /// Get base pointer (for debugging/testing only)
    [[nodiscard]] const uint8_t* base() const noexcept { return base_; }

private:
    /// Round up `value` to the next multiple of `alignment`.
    /// alignment must be a power of 2.
    static constexpr size_t align_up(size_t value, size_t alignment) noexcept {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    uint8_t* base_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
    size_t peak_offset_ = 0;
    size_t allocation_count_ = 0;
};

/// Growable arena that allocates new blocks when the current one is exhausted.
/// Useful when inference memory requirements are unpredictable (variable
/// sequence lengths, dynamic batching).
///
/// Trade-off vs fixed Arena:
///   - Fixed: O(1) alloc, predictable, but fails on overflow
///   - Growable: O(1) amortized alloc, handles variable workloads, slightly
///     worse locality when crossing block boundaries
class GrowableArena {
public:
    explicit GrowableArena(size_t block_size = kDefaultArenaCapacity)
        : block_size_(block_size) {
        add_block();
    }

    template <typename T>
    [[nodiscard]] T* alloc(size_t count, size_t alignment = alignof(T)) {
        size_t bytes = count * sizeof(T);
        return static_cast<T*>(alloc_raw(bytes, alignment));
    }

    [[nodiscard]] void* alloc_raw(size_t bytes, size_t alignment = kCacheLineSize) {
        // Try current block first
        void* ptr = blocks_.back().alloc_raw(bytes, alignment);
        if (ptr) return ptr;

        // Current block exhausted — allocate a new one
        // If requested size > block_size, make a custom-sized block
        size_t new_block_size = std::max(block_size_, bytes + alignment);
        blocks_.emplace_back(new_block_size);
        ++overflow_count_;

        ptr = blocks_.back().alloc_raw(bytes, alignment);
        assert(ptr && "Fresh block should always satisfy allocation");
        return ptr;
    }

    /// Reset all blocks — reuse memory for next inference
    void reset() noexcept {
        // Keep only the first block, reset it
        if (blocks_.size() > 1) {
            Arena first = std::move(blocks_[0]);
            blocks_.clear();
            blocks_.push_back(std::move(first));
        }
        blocks_[0].reset();
    }

    // ─── Stats ───────────────────────────────────────────────────────────────

    [[nodiscard]] size_t total_capacity() const noexcept {
        size_t total = 0;
        for (const auto& b : blocks_) total += b.capacity();
        return total;
    }

    [[nodiscard]] size_t total_used() const noexcept {
        size_t total = 0;
        for (const auto& b : blocks_) total += b.used();
        return total;
    }

    [[nodiscard]] size_t block_count() const noexcept { return blocks_.size(); }
    [[nodiscard]] size_t overflow_count() const noexcept { return overflow_count_; }

private:
    void add_block() {
        blocks_.emplace_back(block_size_);
    }

    size_t block_size_;
    std::vector<Arena> blocks_;
    size_t overflow_count_ = 0;
};

} // namespace inferx::memory
