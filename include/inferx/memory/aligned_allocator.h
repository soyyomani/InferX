#pragma once

/// @file aligned_allocator.h
/// @brief Cache-line and SIMD-aligned memory allocation utilities.
///
/// Why alignment matters for AI workloads:
/// ────────────────────────────────────────
///
/// 1. Cache lines (64 bytes):
///    CPUs fetch memory in 64-byte cache lines. If a tensor starts at a
///    non-aligned address, accessing the first element may require loading
///    TWO cache lines (split load). With alignment, every tensor starts at
///    a cache line boundary — no splits, no wasted bandwidth.
///
/// 2. SIMD registers (16-32 bytes):
///    - ARM NEON: 128-bit (16 bytes) — vld1q_f32 requires 16-byte alignment
///      for best performance (unaligned loads work but may be slower)
///    - x86 AVX: 256-bit (32 bytes) — vmovaps requires 32-byte alignment
///    - x86 AVX-512: 512-bit (64 bytes) — aligned loads are mandatory
///
/// 3. Page alignment (4096 bytes):
///    For very large tensors, page-aligned allocations enable:
///    - mmap-based allocation (direct from OS, no malloc overhead)
///    - Huge pages (2 MB) for reduced TLB misses
///    - DMA transfers (GPU, NIC) require page-aligned buffers
///
/// This file provides:
///   - aligned_alloc/aligned_free: raw allocation primitives
///   - AlignedAllocator<T>: STL-compatible allocator for std::vector
///   - UniqueAligned<T>: RAII smart pointer for aligned memory
///   - Compile-time alignment checks

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>

namespace inferx::memory {

// ─── Alignment Constants ─────────────────────────────────────────────────────

/// SIMD alignment for ARM NEON (128-bit registers)
inline constexpr size_t kNeonAlignment = 16;

/// SIMD alignment for x86 AVX (256-bit registers)
inline constexpr size_t kAvxAlignment = 32;

/// Cache line alignment (Intel, AMD, Apple M-series)
inline constexpr size_t kCacheLineAlignment = 64;

/// AVX-512 alignment
inline constexpr size_t kAvx512Alignment = 64;

/// Page alignment (standard 4 KB pages)
inline constexpr size_t kPageAlignment = 4096;

/// Default tensor alignment: cache line (covers all SIMD widths ≤ 64 bytes)
inline constexpr size_t kTensorAlignment = kCacheLineAlignment;

// ─── Compile-Time Utilities ──────────────────────────────────────────────────

/// Check if a value is a power of 2 (required for alignment)
constexpr bool is_power_of_2(size_t n) noexcept {
    return n > 0 && (n & (n - 1)) == 0;
}

/// Round up to alignment boundary (compile-time capable)
constexpr size_t align_up(size_t value, size_t alignment) noexcept {
    return (value + alignment - 1) & ~(alignment - 1);
}

/// Check if a pointer is aligned to the given boundary
inline bool is_aligned(const void* ptr, size_t alignment) noexcept {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// ─── Raw Allocation Functions ────────────────────────────────────────────────

/// Allocate `size` bytes with `alignment`-byte alignment.
/// Returns nullptr on failure.
///
/// Platform notes:
///   - macOS/Linux: std::aligned_alloc (C11) — size must be multiple of alignment
///   - Windows: _aligned_malloc (different free function)
///   - We normalize by rounding size up to alignment multiple
[[nodiscard]] inline void* aligned_malloc(size_t size, size_t alignment = kTensorAlignment) {
    static_assert(sizeof(void*) >= 4, "Unsupported platform");

    if (size == 0) return nullptr;
    if (!is_power_of_2(alignment)) return nullptr;

    // std::aligned_alloc requires size to be a multiple of alignment
    size_t aligned_size = align_up(size, alignment);
    return std::aligned_alloc(alignment, aligned_size);
}

/// Free memory allocated with aligned_malloc.
/// Safe to call with nullptr.
inline void aligned_free(void* ptr) noexcept {
    std::free(ptr); // std::aligned_alloc memory is freed with std::free
}

// ─── RAII Smart Pointer ──────────────────────────────────────────────────────

/// Custom deleter for aligned memory
struct AlignedDeleter {
    void operator()(void* ptr) const noexcept {
        aligned_free(ptr);
    }
};

/// RAII wrapper for aligned memory — like unique_ptr but for aligned allocations.
/// Automatically frees on destruction, supports move semantics.
///
/// Usage:
///   auto buf = make_aligned<float>(1024);  // 1024 floats, 64-byte aligned
///   float* ptr = buf.get();
///   // ... use ptr ...
///   // automatically freed when buf goes out of scope
template <typename T>
using UniqueAligned = std::unique_ptr<T[], AlignedDeleter>;

/// Create an aligned buffer for `count` elements of type T.
/// Memory is NOT initialized (like malloc, not calloc).
template <typename T>
[[nodiscard]] UniqueAligned<T> make_aligned(size_t count, size_t alignment = kTensorAlignment) {
    void* ptr = aligned_malloc(count * sizeof(T), alignment);
    if (!ptr && count > 0) {
        throw std::bad_alloc();
    }
    return UniqueAligned<T>(static_cast<T*>(ptr));
}

/// Create an aligned buffer initialized to zero.
template <typename T>
[[nodiscard]] UniqueAligned<T> make_aligned_zero(size_t count, size_t alignment = kTensorAlignment) {
    auto buf = make_aligned<T>(count, alignment);
    std::memset(buf.get(), 0, count * sizeof(T));
    return buf;
}

// ─── STL-Compatible Allocator ────────────────────────────────────────────────

/// STL allocator that produces aligned memory. Use with std::vector:
///
///   std::vector<float, AlignedAllocator<float>> data(1024);
///   assert(is_aligned(data.data(), 64));  // guaranteed!
///
/// This lets you use standard containers while maintaining SIMD alignment.
/// Useful for weight matrices that live in std::vector but need aligned access.
template <typename T, size_t Alignment = kTensorAlignment>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    static_assert(is_power_of_2(Alignment), "Alignment must be a power of 2");
    static_assert(Alignment >= alignof(T), "Alignment must be >= alignof(T)");

    constexpr AlignedAllocator() noexcept = default;

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(size_type n) {
        if (n == 0) return nullptr;

        size_t bytes = n * sizeof(T);
        void* ptr = aligned_malloc(bytes, Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_type /*n*/) noexcept {
        aligned_free(ptr);
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

/// Equality operators (required by STL allocator concept)
template <typename T, typename U, size_t A>
constexpr bool operator==(const AlignedAllocator<T, A>&,
                           const AlignedAllocator<U, A>&) noexcept {
    return true;
}

template <typename T, typename U, size_t A>
constexpr bool operator!=(const AlignedAllocator<T, A>&,
                           const AlignedAllocator<U, A>&) noexcept {
    return false;
}

// ─── Type Aliases for Common Use Cases ───────────────────────────────────────

/// Aligned float vector — use for weight matrices, activation buffers
using AlignedFloatVec = std::vector<float, AlignedAllocator<float>>;

/// Aligned int8 vector — use for quantized weights
using AlignedInt8Vec = std::vector<int8_t, AlignedAllocator<int8_t>>;

/// Aligned uint8 vector — use for image data
using AlignedUint8Vec = std::vector<uint8_t, AlignedAllocator<uint8_t>>;

} // namespace inferx::memory
