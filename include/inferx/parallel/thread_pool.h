#pragma once

/// @file thread_pool.h
/// @brief Fixed-size thread pool with task queue for parallel execution.
///
/// Why a thread pool for inference?
/// ─────────────────────────────────
/// AI inference has two types of parallelism:
///
///   1. Inter-op parallelism: independent graph nodes run concurrently
///      Example: Q, K, V projections in attention are independent → 3× speedup
///
///   2. Intra-op parallelism: a single operation splits across threads
///      Example: MatMul rows partitioned across 8 cores → 8× speedup
///
/// Without a thread pool:
///   - std::thread creation: ~20-50 µs per thread (OS syscall)
///   - For 100 inference calls/sec × 8 threads = 800 thread creates/sec
///   - Total overhead: 40 ms/sec wasted just on thread management
///
/// With a thread pool:
///   - Threads created ONCE at startup
///   - Task dispatch: ~50 ns (push to queue + notify)
///   - 800× less overhead than thread-per-task
///
/// This is the same pattern used by:
///   - oneDNN: dnnl_thread_pool for parallel kernels
///   - TensorRT: enqueue() dispatches to internal thread pool
///   - ONNX Runtime: ThreadPool for inter/intra-op parallelism
///   - Eigen: ThreadPoolDevice for TensorFlow CPU backend
///
/// Design:
///   - Fixed N worker threads (default: hardware_concurrency)
///   - Lock-based MPSC queue (multiple producers, single consumer per thread)
///   - Condition variable for sleep/wake (no busy spinning = power efficient)
///   - Graceful shutdown: finish pending tasks, then join threads

#include <cstddef>
#include <cstdint>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <vector>
#include <atomic>
#include <type_traits>
#include <memory>

namespace inferx::parallel {

/// Thread pool: manages N worker threads that execute submitted tasks.
///
/// Usage:
///   ThreadPool pool(8);  // 8 worker threads
///   auto future = pool.submit([] { return heavy_computation(); });
///   auto result = future.get();  // blocks until done
///
///   pool.submit([&] { matmul_rows(A, B, C, 0, M/2); });  // fire-and-forget
///   pool.submit([&] { matmul_rows(A, B, C, M/2, M); });
///   pool.wait_idle();  // wait for both to complete
class ThreadPool {
public:
    /// Create thread pool with specified number of workers.
    /// Default: hardware_concurrency() (all available cores).
    explicit ThreadPool(size_t num_threads = 0)
        : stop_(false), active_tasks_(0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4; // Fallback
        }

        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    /// Destructor: signals shutdown, waits for all workers to finish.
    ~ThreadPool() {
        shutdown();
    }

    // Non-copyable, non-movable (owns threads)
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    /// Submit a callable for asynchronous execution.
    /// Returns a future to retrieve the result.
    ///
    /// This is the primary API. Works with any callable:
    ///   pool.submit([] { return 42; });
    ///   pool.submit([&] { process(data); });
    ///   pool.submit(std::bind(&Obj::method, &obj, args...));
    template <typename F, typename... Args>
    auto submit(F&& func, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {

        using ReturnType = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(func), std::forward<Args>(args)...)
        );

        std::future<ReturnType> future = task->get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                throw std::runtime_error("ThreadPool: submit after shutdown");
            }
            tasks_.emplace([task, this] {
                active_tasks_.fetch_add(1, std::memory_order_relaxed);
                (*task)();
                active_tasks_.fetch_sub(1, std::memory_order_relaxed);
                idle_cv_.notify_all();
            });
        }

        cv_.notify_one();
        return future;
    }

    /// Submit a void task (fire-and-forget convenience).
    void enqueue(std::function<void()> func) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) return;
            tasks_.emplace([f = std::move(func), this] {
                active_tasks_.fetch_add(1, std::memory_order_relaxed);
                f();
                active_tasks_.fetch_sub(1, std::memory_order_relaxed);
                idle_cv_.notify_all();
            });
        }
        cv_.notify_one();
    }

    /// Wait until all submitted tasks have completed.
    /// Does NOT prevent new submissions — just waits for the queue to drain.
    void wait_idle() {
        std::unique_lock<std::mutex> lock(idle_mutex_);
        idle_cv_.wait(lock, [this] {
            return tasks_empty() && active_tasks_.load(std::memory_order_relaxed) == 0;
        });
    }

    /// Graceful shutdown: stop accepting new tasks, finish pending, join threads.
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) return;
            stop_ = true;
        }
        cv_.notify_all();

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // ─── Accessors ───────────────────────────────────────────────────────────

    /// Number of worker threads
    [[nodiscard]] size_t num_threads() const noexcept { return workers_.size(); }

    /// Number of tasks waiting in the queue
    [[nodiscard]] size_t pending_tasks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }

    /// Number of tasks currently being executed
    [[nodiscard]] size_t active_count() const noexcept {
        return active_tasks_.load(std::memory_order_relaxed);
    }

    /// Whether the pool has been shut down
    [[nodiscard]] bool is_stopped() const noexcept { return stop_; }

private:
    /// Worker thread main loop: wait for tasks, execute them.
    void worker_loop() {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

                if (stop_ && tasks_.empty()) {
                    return; // Shutdown: no more tasks
                }

                task = std::move(tasks_.front());
                tasks_.pop();
            }

            task(); // Execute outside the lock
        }
    }

    /// Check if task queue is empty (must hold mutex_)
    bool tasks_empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.empty();
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    mutable std::mutex mutex_;              ///< Protects tasks_ and stop_
    std::condition_variable cv_;            ///< Wakes workers when tasks available

    std::mutex idle_mutex_;                 ///< Protects idle wait
    std::condition_variable idle_cv_;       ///< Notified when a task completes

    std::atomic<bool> stop_;                ///< Shutdown flag
    std::atomic<size_t> active_tasks_;      ///< Tasks currently executing
};

} // namespace inferx::parallel
