/// @file test_memory.cpp
/// @brief Correctness tests for InferX memory management module.
///
/// Tests cover:
///   - Arena: allocation, alignment, reset, overflow, ownership, growable
///   - BufferPool: acquire/release, reuse, eviction, stats, drain
///   - AlignedAllocator: alignment guarantees, STL compatibility, RAII
///   - MemoryStats: formatting, profiler tracking

#include <gtest/gtest.h>
#include <inferx/memory/arena.h>
#include <inferx/memory/buffer_pool.h>
#include <inferx/memory/aligned_allocator.h>
#include <inferx/memory/memory_stats.h>

#include <vector>
#include <cstdint>
#include <numeric>

using namespace inferx::memory;

// ═══════════════════════════════════════════════════════════════════════════════
// Arena Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Arena, BasicAllocation) {
    Arena arena(4096);
    float* ptr = arena.alloc<float>(100);
    ASSERT_NE(ptr, nullptr);
    // Write and read back
    for (int i = 0; i < 100; ++i) ptr[i] = static_cast<float>(i);
    for (int i = 0; i < 100; ++i) EXPECT_EQ(ptr[i], static_cast<float>(i));
}

TEST(Arena, MultipleAllocations) {
    Arena arena(4096);
    float* a = arena.alloc<float>(10);
    float* b = arena.alloc<float>(20);
    float* c = arena.alloc<float>(30);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);
    // All pointers should be different
    EXPECT_NE(a, b);
    EXPECT_NE(b, c);
    EXPECT_NE(a, c);
}

TEST(Arena, AlignmentGuaranteed) {
    Arena arena(8192);
    // Request with 64-byte alignment
    void* ptr1 = arena.alloc_raw(100, 64);
    void* ptr2 = arena.alloc_raw(37, 64);  // Odd size
    void* ptr3 = arena.alloc_raw(1, 64);   // Tiny

    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    ASSERT_NE(ptr3, nullptr);

    EXPECT_TRUE(is_aligned(ptr1, 64));
    EXPECT_TRUE(is_aligned(ptr2, 64));
    EXPECT_TRUE(is_aligned(ptr3, 64));
}

TEST(Arena, ResetReusesMemory) {
    Arena arena(4096);
    float* first = arena.alloc<float>(100);
    size_t used_before_reset = arena.used();
    EXPECT_GT(used_before_reset, 0u);

    arena.reset();
    EXPECT_EQ(arena.used(), 0u);

    // After reset, next allocation starts from the beginning
    float* second = arena.alloc<float>(100);
    ASSERT_NE(second, nullptr);
    // They should be at the same offset (same alignment)
    EXPECT_EQ(first, second);
}

TEST(Arena, OverflowReturnsNull) {
    Arena arena(256);  // Small arena
    // Try to allocate more than capacity
    float* ptr = arena.alloc<float>(1000);  // 4000 bytes > 256
    EXPECT_EQ(ptr, nullptr);
}

TEST(Arena, OwnershipTracking) {
    Arena arena(4096);
    float* inside = arena.alloc<float>(10);
    float outside_data[10];

    EXPECT_TRUE(arena.owns(inside));
    EXPECT_FALSE(arena.owns(outside_data));
    EXPECT_FALSE(arena.owns(nullptr));
}

TEST(Arena, PeakUsageTracking) {
    Arena arena(4096);
    arena.alloc<float>(100);  // 400 bytes + alignment
    size_t peak1 = arena.peak_usage();

    arena.reset();
    EXPECT_EQ(arena.used(), 0u);
    EXPECT_EQ(arena.peak_usage(), peak1);  // Peak preserved across reset

    arena.alloc<float>(200);  // More than before
    EXPECT_GT(arena.peak_usage(), peak1);  // New peak
}

TEST(Arena, AllocationCount) {
    Arena arena(4096);
    EXPECT_EQ(arena.allocation_count(), 0u);
    arena.alloc<float>(10);
    EXPECT_EQ(arena.allocation_count(), 1u);
    arena.alloc<float>(20);
    EXPECT_EQ(arena.allocation_count(), 2u);
    arena.reset();
    EXPECT_EQ(arena.allocation_count(), 2u);  // Count persists
}

TEST(Arena, Remaining) {
    Arena arena(1024);
    size_t initial_remaining = arena.remaining();
    EXPECT_GE(initial_remaining, 1024u);  // May be slightly more due to align_up

    arena.alloc_raw(512, 1);
    EXPECT_LT(arena.remaining(), initial_remaining);
    EXPECT_GE(arena.remaining(), 512u);
}

TEST(Arena, MoveSemantics) {
    Arena arena1(4096);
    float* ptr = arena1.alloc<float>(10);
    ASSERT_NE(ptr, nullptr);

    Arena arena2 = std::move(arena1);
    EXPECT_TRUE(arena2.owns(ptr));
    EXPECT_GT(arena2.used(), 0u);
}

TEST(Arena, ZeroCapacityThrows) {
    EXPECT_THROW(Arena(0), std::invalid_argument);
}

// ─── GrowableArena ───────────────────────────────────────────────────────────

TEST(GrowableArena, BasicAllocation) {
    GrowableArena arena(1024);
    float* ptr = arena.alloc<float>(100);
    ASSERT_NE(ptr, nullptr);
    for (int i = 0; i < 100; ++i) ptr[i] = static_cast<float>(i * 2);
    for (int i = 0; i < 100; ++i) EXPECT_EQ(ptr[i], static_cast<float>(i * 2));
}

TEST(GrowableArena, GrowsOnOverflow) {
    GrowableArena arena(256);  // Tiny first block
    EXPECT_EQ(arena.block_count(), 1u);

    // Allocate more than 256 bytes
    float* ptr = arena.alloc<float>(100);  // 400 bytes > 256
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(arena.block_count(), 2u);  // Grew
    EXPECT_EQ(arena.overflow_count(), 1u);
}

TEST(GrowableArena, ResetKeepsFirstBlock) {
    GrowableArena arena(1024);
    arena.alloc<float>(1000);  // Force overflow
    EXPECT_GT(arena.block_count(), 1u);

    arena.reset();
    EXPECT_EQ(arena.block_count(), 1u);  // Shrinks back to one block
}

// ═══════════════════════════════════════════════════════════════════════════════
// BufferPool Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(BufferPool, AcquireAndRelease) {
    BufferPool pool;
    auto buf = pool.acquire(1024);
    ASSERT_NE(buf.data, nullptr);
    EXPECT_GE(buf.capacity, 1024u);

    // Write to verify usability
    auto* ptr = static_cast<float*>(buf.data);
    ptr[0] = 42.0f;
    EXPECT_EQ(ptr[0], 42.0f);

    pool.release(std::move(buf));
}

TEST(BufferPool, ReuseOnSecondAcquire) {
    BufferPool pool;

    // First acquire — cache miss (fresh allocation)
    auto buf1 = pool.acquire(1024);
    void* original_ptr = buf1.data;
    pool.release(std::move(buf1));

    // Second acquire — should be a cache hit (same buffer reused)
    auto buf2 = pool.acquire(1024);
    EXPECT_EQ(buf2.data, original_ptr);  // Same pointer!

    pool.release(std::move(buf2));
}

TEST(BufferPool, StatsTracking) {
    BufferPool pool;

    auto buf1 = pool.acquire(512);
    EXPECT_EQ(pool.stats().misses, 1u);
    EXPECT_EQ(pool.stats().hits, 0u);
    pool.release(std::move(buf1));

    auto buf2 = pool.acquire(512);
    EXPECT_EQ(pool.stats().misses, 1u);
    EXPECT_EQ(pool.stats().hits, 1u);
    pool.release(std::move(buf2));

    EXPECT_DOUBLE_EQ(pool.stats().hit_rate(), 0.5);
}

TEST(BufferPool, EvictionWhenFull) {
    BufferPool pool(2);  // Max 2 buffers per bucket

    // Acquire and release 3 buffers of same size
    auto b1 = pool.acquire(1024);
    auto b2 = pool.acquire(1024);
    auto b3 = pool.acquire(1024);
    pool.release(std::move(b1));
    pool.release(std::move(b2));
    pool.release(std::move(b3));  // This one should be evicted

    EXPECT_EQ(pool.cached_count(), 2u);
    EXPECT_EQ(pool.stats().evictions, 1u);
}

TEST(BufferPool, DifferentSizesUseDifferentBuckets) {
    BufferPool pool;

    auto small = pool.acquire(100);    // Bucket: 128
    auto medium = pool.acquire(1000);  // Bucket: 1024
    auto large = pool.acquire(5000);   // Bucket: 8192

    pool.release(std::move(small));
    pool.release(std::move(medium));
    pool.release(std::move(large));

    EXPECT_EQ(pool.bucket_count(), 3u);
}

TEST(BufferPool, DrainFreesAll) {
    BufferPool pool;

    auto buf = pool.acquire(4096);
    pool.release(std::move(buf));
    EXPECT_EQ(pool.cached_count(), 1u);

    pool.drain();
    EXPECT_EQ(pool.cached_count(), 0u);
}

TEST(BufferPool, TypedAcquire) {
    BufferPool pool;
    auto buf = pool.acquire_typed<float>(256);  // 256 floats = 1024 bytes
    ASSERT_NE(buf.data, nullptr);
    EXPECT_GE(buf.capacity, 256 * sizeof(float));

    auto* ptr = static_cast<float*>(buf.data);
    ptr[255] = 99.0f;
    EXPECT_EQ(ptr[255], 99.0f);

    pool.release(std::move(buf));
}

TEST(BufferPool, HitRateAfterWarmup) {
    BufferPool pool;

    // Simulate inference loop: same sizes every iteration
    for (int iter = 0; iter < 100; ++iter) {
        auto q = pool.acquire(768 * 4);    // Q projection
        auto k = pool.acquire(768 * 4);    // K projection
        auto v = pool.acquire(768 * 4);    // V projection
        pool.release(std::move(q));
        pool.release(std::move(k));
        pool.release(std::move(v));
    }

    // After first iteration (3 misses), all subsequent are hits
    // Hit rate: 297 / 300 = 99%
    EXPECT_GT(pool.stats().hit_rate(), 0.95);
}

// ═══════════════════════════════════════════════════════════════════════════════
// AlignedAllocator Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(AlignedAllocator, RawAllocationAlignment) {
    void* ptr = aligned_malloc(1024, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(is_aligned(ptr, 64));
    aligned_free(ptr);
}

TEST(AlignedAllocator, PageAlignment) {
    void* ptr = aligned_malloc(8192, kPageAlignment);
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(is_aligned(ptr, 4096));
    aligned_free(ptr);
}

TEST(AlignedAllocator, UniqueAlignedRAII) {
    auto buf = make_aligned<float>(1024);
    ASSERT_NE(buf.get(), nullptr);
    EXPECT_TRUE(is_aligned(buf.get(), kTensorAlignment));

    // Write/read test
    buf[0] = 1.0f;
    buf[1023] = 99.0f;
    EXPECT_EQ(buf[0], 1.0f);
    EXPECT_EQ(buf[1023], 99.0f);
    // RAII: automatically freed when buf goes out of scope
}

TEST(AlignedAllocator, ZeroInitialization) {
    auto buf = make_aligned_zero<float>(512);
    ASSERT_NE(buf.get(), nullptr);
    for (size_t i = 0; i < 512; ++i) {
        EXPECT_EQ(buf[i], 0.0f);
    }
}

TEST(AlignedAllocator, STLVectorAligned) {
    AlignedFloatVec vec(1024);
    EXPECT_TRUE(is_aligned(vec.data(), kTensorAlignment));

    // Standard vector operations work
    std::iota(vec.begin(), vec.end(), 0.0f);
    EXPECT_EQ(vec[0], 0.0f);
    EXPECT_EQ(vec[1023], 1023.0f);

    // Push back works
    vec.push_back(42.0f);
    EXPECT_EQ(vec.back(), 42.0f);
    // After reallocation, alignment should still hold
    EXPECT_TRUE(is_aligned(vec.data(), kTensorAlignment));
}

TEST(AlignedAllocator, PowerOf2Check) {
    EXPECT_TRUE(is_power_of_2(1));
    EXPECT_TRUE(is_power_of_2(2));
    EXPECT_TRUE(is_power_of_2(64));
    EXPECT_TRUE(is_power_of_2(4096));
    EXPECT_FALSE(is_power_of_2(0));
    EXPECT_FALSE(is_power_of_2(3));
    EXPECT_FALSE(is_power_of_2(65));
}

TEST(AlignedAllocator, AlignUpCorrectness) {
    EXPECT_EQ(align_up(0, 64), 0u);
    EXPECT_EQ(align_up(1, 64), 64u);
    EXPECT_EQ(align_up(63, 64), 64u);
    EXPECT_EQ(align_up(64, 64), 64u);
    EXPECT_EQ(align_up(65, 64), 128u);
    EXPECT_EQ(align_up(100, 16), 112u);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MemoryStats Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(MemoryStats, FormatBytes) {
    EXPECT_EQ(format_bytes(0), "0 B");
    EXPECT_EQ(format_bytes(512), "512 B");
    EXPECT_EQ(format_bytes(1024), "1.00 KB");
    EXPECT_EQ(format_bytes(1536), "1.50 KB");
    EXPECT_EQ(format_bytes(1048576), "1.00 MB");
    EXPECT_EQ(format_bytes(1073741824), "1.00 GB");
}

TEST(MemoryStats, FormatPercent) {
    EXPECT_EQ(format_percent(0.0), "0.0%");
    EXPECT_EQ(format_percent(0.5), "50.0%");
    EXPECT_EQ(format_percent(1.0), "100.0%");
    EXPECT_EQ(format_percent(0.875), "87.5%");
}

TEST(MemoryStats, StatsUtilization) {
    MemoryStats stats;
    stats.name = "TestArena";
    stats.total_capacity = 1024;
    stats.peak_used = 768;
    EXPECT_DOUBLE_EQ(stats.utilization(), 0.75);
}

TEST(MemoryStats, StatsFragmentation) {
    MemoryStats stats;
    stats.current_used = 900;
    stats.bytes_wasted = 100;
    EXPECT_DOUBLE_EQ(stats.fragmentation(), 0.1);
}

TEST(MemoryStats, StatsReuseRate) {
    MemoryStats stats;
    stats.allocation_count = 100;
    stats.reuse_count = 75;
    EXPECT_DOUBLE_EQ(stats.reuse_rate(), 0.75);
}

TEST(MemoryStats, ProfilerRecordAlloc) {
    auto& profiler = MemoryProfiler::instance();
    profiler.reset();

    profiler.record_alloc(1024);
    profiler.record_alloc(2048);

    EXPECT_EQ(profiler.total_allocated(), 3072u);
    EXPECT_EQ(profiler.current_usage(), 3072u);
    EXPECT_EQ(profiler.peak_usage(), 3072u);
    EXPECT_EQ(profiler.alloc_count(), 2u);

    profiler.record_free(1024);
    EXPECT_EQ(profiler.current_usage(), 2048u);
    EXPECT_EQ(profiler.peak_usage(), 3072u);  // Peak unchanged
    EXPECT_EQ(profiler.free_count(), 1u);

    profiler.reset();
}

TEST(MemoryStats, SummaryProducesOutput) {
    MemoryStats stats;
    stats.name = "TestComponent";
    stats.total_capacity = 16 * 1024 * 1024;
    stats.current_used = 4 * 1024 * 1024;
    stats.peak_used = 8 * 1024 * 1024;
    stats.allocation_count = 500;
    stats.active_allocations = 10;

    std::string summary = stats.summary();
    EXPECT_FALSE(summary.empty());
    EXPECT_NE(summary.find("TestComponent"), std::string::npos);
    EXPECT_NE(summary.find("16.00 MB"), std::string::npos);
}
