#pragma once

/// @file memory_stats.h
/// @brief Unified memory tracking and reporting for the InferX runtime.
///
/// Why memory profiling matters in AI inference:
/// ──────────────────────────────────────────────
///
/// Memory is the primary bottleneck in modern AI inference:
///   - GPT-2 small: 500 MB weights + ~200 MB activations per request
///   - LLaMA-7B:    14 GB weights (FP16) + ~2 GB KV cache per request
///   - ResNet-50:   100 MB weights + ~50 MB activations
///
/// Without profiling, you can't answer critical deployment questions:
///   - "What's our peak memory during inference?" (capacity planning)
///   - "How much memory is wasted on fragmentation?" (allocator tuning)
///   - "What's the allocation rate on the hot path?" (latency debugging)
///   - "Can we batch 2 more requests?" (throughput optimization)
///
/// This module provides:
///   1. MemoryStats: per-component tracking (arena, pool, global)
///   2. MemoryProfiler: singleton that aggregates stats across all allocators
///   3. ScopedMemoryTracker: RAII helper for measuring a code region's memory
///   4. Human-readable reporting (bytes → KB/MB/GB)
///
/// Real-world parallels:
///   - NVIDIA: nvidia-smi memory tracking + TensorRT memory profiler
///   - PyTorch: torch.cuda.memory_stats() / torch.cuda.memory_summary()
///   - TensorFlow: tf.config.experimental.get_memory_info()

#include <cstddef>
#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <atomic>

namespace inferx::memory {

// ─── Formatting Utilities ────────────────────────────────────────────────────

/// Format bytes into human-readable string (e.g., "1.50 MB")
inline std::string format_bytes(size_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    if (bytes >= 1024ULL * 1024 * 1024) {
        oss << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) << " GB";
    } else if (bytes >= 1024ULL * 1024) {
        oss << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MB";
    } else if (bytes >= 1024) {
        oss << static_cast<double>(bytes) / 1024.0 << " KB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

/// Format a ratio as percentage string (e.g., "87.5%")
inline std::string format_percent(double ratio) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << (ratio * 100.0) << "%";
    return oss.str();
}

// ─── Memory Statistics ───────────────────────────────────────────────────────

/// Snapshot of memory usage for a single component (arena, pool, etc.)
struct MemoryStats {
    std::string name;                   ///< Component name (e.g., "ArenaMain")

    // Capacity
    size_t total_capacity = 0;          ///< Total memory reserved from OS
    size_t current_used = 0;            ///< Currently in-use bytes
    size_t peak_used = 0;               ///< High-water mark

    // Allocation counts
    size_t allocation_count = 0;        ///< Total allocations performed
    size_t deallocation_count = 0;      ///< Total frees/releases
    size_t active_allocations = 0;      ///< Currently live allocations

    // Efficiency
    size_t bytes_wasted = 0;            ///< Alignment padding / bucket rounding waste
    size_t reuse_count = 0;             ///< Times a buffer was reused (pool hits)

    // ─── Derived Metrics ─────────────────────────────────────────────────────

    /// Memory utilization: peak_used / total_capacity
    [[nodiscard]] double utilization() const noexcept {
        return total_capacity > 0
            ? static_cast<double>(peak_used) / static_cast<double>(total_capacity)
            : 0.0;
    }

    /// Fragmentation estimate: wasted / (used + wasted)
    [[nodiscard]] double fragmentation() const noexcept {
        size_t total = current_used + bytes_wasted;
        return total > 0
            ? static_cast<double>(bytes_wasted) / static_cast<double>(total)
            : 0.0;
    }

    /// Reuse rate: reuse_count / allocation_count
    [[nodiscard]] double reuse_rate() const noexcept {
        return allocation_count > 0
            ? static_cast<double>(reuse_count) / static_cast<double>(allocation_count)
            : 0.0;
    }

    /// Generate a human-readable summary
    [[nodiscard]] std::string summary() const {
        std::ostringstream oss;
        oss << "┌─ " << name << " ─────────────────────────────\n"
            << "│ Capacity:     " << format_bytes(total_capacity) << "\n"
            << "│ Current Used: " << format_bytes(current_used) << "\n"
            << "│ Peak Used:    " << format_bytes(peak_used) << "\n"
            << "│ Utilization:  " << format_percent(utilization()) << "\n"
            << "│ Allocations:  " << allocation_count << "\n"
            << "│ Active:       " << active_allocations << "\n"
            << "│ Reuse Rate:   " << format_percent(reuse_rate()) << "\n"
            << "│ Fragmentation:" << format_percent(fragmentation()) << "\n"
            << "└──────────────────────────────────────────\n";
        return oss.str();
    }
};

// ─── Memory Profiler ─────────────────────────────────────────────────────────

/// Global memory profiler that tracks all allocator instances.
/// Provides a unified view of memory usage across the runtime.
///
/// Usage:
///   MemoryProfiler::instance().register_stats("arena_main", stats);
///   std::cout << MemoryProfiler::instance().report();
class MemoryProfiler {
public:
    static MemoryProfiler& instance() {
        static MemoryProfiler profiler;
        return profiler;
    }

    /// Record a memory event (allocation or free)
    void record_alloc(size_t bytes) noexcept {
        total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
        alloc_count_.fetch_add(1, std::memory_order_relaxed);

        // Update peak (relaxed is fine — this is approximate profiling)
        size_t current = current_usage_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        size_t peak = peak_usage_.load(std::memory_order_relaxed);
        while (current > peak) {
            if (peak_usage_.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {
                break;
            }
        }
    }

    void record_free(size_t bytes) noexcept {
        current_usage_.fetch_sub(bytes, std::memory_order_relaxed);
        free_count_.fetch_add(1, std::memory_order_relaxed);
    }

    /// Register a named stats snapshot (e.g., from an arena or pool)
    void register_stats(MemoryStats stats) {
        snapshots_.push_back(std::move(stats));
    }

    /// Clear all registered snapshots
    void clear_snapshots() { snapshots_.clear(); }

    // ─── Accessors ───────────────────────────────────────────────────────────

    [[nodiscard]] size_t total_allocated() const noexcept {
        return total_allocated_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] size_t current_usage() const noexcept {
        return current_usage_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] size_t peak_usage() const noexcept {
        return peak_usage_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] size_t alloc_count() const noexcept {
        return alloc_count_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] size_t free_count() const noexcept {
        return free_count_.load(std::memory_order_relaxed);
    }

    /// Generate a full report of all memory subsystems
    [[nodiscard]] std::string report() const {
        std::ostringstream oss;
        oss << "╔══════════════════════════════════════════════╗\n"
            << "║       InferX Memory Profiler Report          ║\n"
            << "╠══════════════════════════════════════════════╣\n"
            << "║ Global Stats:                                ║\n"
            << "║   Total Allocated: " << std::setw(20) << format_bytes(total_allocated()) << "   ║\n"
            << "║   Current Usage:   " << std::setw(20) << format_bytes(current_usage()) << "   ║\n"
            << "║   Peak Usage:      " << std::setw(20) << format_bytes(peak_usage()) << "   ║\n"
            << "║   Alloc Count:     " << std::setw(20) << alloc_count() << "   ║\n"
            << "║   Free Count:      " << std::setw(20) << free_count() << "   ║\n"
            << "╚══════════════════════════════════════════════╝\n\n";

        for (const auto& snap : snapshots_) {
            oss << snap.summary() << "\n";
        }

        return oss.str();
    }

    /// Reset all counters
    void reset() noexcept {
        total_allocated_.store(0, std::memory_order_relaxed);
        current_usage_.store(0, std::memory_order_relaxed);
        peak_usage_.store(0, std::memory_order_relaxed);
        alloc_count_.store(0, std::memory_order_relaxed);
        free_count_.store(0, std::memory_order_relaxed);
        snapshots_.clear();
    }

private:
    MemoryProfiler() = default;

    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> current_usage_{0};
    std::atomic<size_t> peak_usage_{0};
    std::atomic<size_t> alloc_count_{0};
    std::atomic<size_t> free_count_{0};

    std::vector<MemoryStats> snapshots_;
};

// ─── Scoped Memory Tracker ───────────────────────────────────────────────────

/// RAII helper that measures memory usage of a code region.
///
/// Usage:
///   {
///       ScopedMemoryTracker tracker("attention_layer");
///       // ... allocate tensors, do compute ...
///   }
///   // tracker.peak_bytes() available after scope exits
///
/// This is the C++ equivalent of:
///   torch.cuda.reset_peak_memory_stats()
///   # ... code ...
///   peak = torch.cuda.max_memory_allocated()
class ScopedMemoryTracker {
public:
    explicit ScopedMemoryTracker(std::string name = "unnamed")
        : name_(std::move(name)),
          start_usage_(MemoryProfiler::instance().current_usage()),
          start_time_(std::chrono::steady_clock::now()) {}

    ~ScopedMemoryTracker() {
        end_usage_ = MemoryProfiler::instance().current_usage();
        end_time_ = std::chrono::steady_clock::now();
    }

    /// Name of the tracked region
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    /// Memory delta: end_usage - start_usage (may be negative if memory was freed)
    [[nodiscard]] int64_t memory_delta() const noexcept {
        return static_cast<int64_t>(end_usage_) - static_cast<int64_t>(start_usage_);
    }

    /// Duration of the tracked region
    [[nodiscard]] double elapsed_ms() const noexcept {
        auto duration = end_time_ - start_time_;
        return std::chrono::duration<double, std::milli>(duration).count();
    }

    /// Starting memory level
    [[nodiscard]] size_t start_usage() const noexcept { return start_usage_; }

    /// Ending memory level
    [[nodiscard]] size_t end_usage() const noexcept { return end_usage_; }

private:
    std::string name_;
    size_t start_usage_;
    size_t end_usage_ = 0;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point end_time_;
};

} // namespace inferx::memory
