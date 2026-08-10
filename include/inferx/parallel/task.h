#pragma once

/// @file task.h
/// @brief Task abstraction with dependencies for graph-level parallelism.
///
/// In computational graph execution, not all operations can run immediately:
///   - MatMul_Q depends on Input and W_Q being ready
///   - ReLU depends on MatMul completing
///   - But MatMul_Q, MatMul_K, MatMul_V are independent → run in parallel!
///
/// A Task wraps a callable with:
///   - A list of predecessor task IDs it depends on
///   - A state machine (Pending → Ready → Running → Done)
///   - An atomic dependency counter (decremented by predecessors)
///   - When counter reaches 0 → task becomes Ready → submitted to pool
///
/// This is the "task graph" execution model used by:
///   - TBB (Intel Threading Building Blocks): tbb::flow::graph
///   - taskflow: tf::Taskflow with dependency DAG
///   - CUDA Graphs: cudaGraphAddDependencies
///   - Vulkan Compute: semaphore-based dependency chains
///
/// Usage:
///   TaskGraph graph;
///   auto t1 = graph.add_task([] { compute_Q(); });
///   auto t2 = graph.add_task([] { compute_K(); });
///   auto t3 = graph.add_task([] { compute_V(); });
///   auto t4 = graph.add_task([] { attention(Q, K, V); }, {t1, t2, t3});
///   graph.execute(pool);  // t1,t2,t3 run in parallel, t4 waits for all three

#include <inferx/parallel/thread_pool.h>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <functional>
#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace inferx::parallel {

/// Task state machine
enum class TaskState {
    Pending,    ///< Waiting for dependencies
    Ready,      ///< All dependencies met, queued for execution
    Running,    ///< Currently executing on a worker thread
    Done,       ///< Execution complete
    Failed      ///< Execution threw an exception
};

/// A single task in the task graph.
class Task {
public:
    using TaskId = size_t;
    using TaskFunc = std::function<void()>;

    Task(TaskId id, TaskFunc func, std::string name = "")
        : id_(id), func_(std::move(func)),
          name_(name.empty() ? "task_" + std::to_string(id) : std::move(name)),
          state_(TaskState::Pending), pending_deps_(0) {}

    // ─── Accessors ───────────────────────────────────────────────────────────

    [[nodiscard]] TaskId id() const noexcept { return id_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] TaskState state() const noexcept { return state_.load(std::memory_order_acquire); }

    [[nodiscard]] const std::vector<TaskId>& predecessors() const noexcept { return predecessors_; }
    [[nodiscard]] const std::vector<TaskId>& successors() const noexcept { return successors_; }

    // ─── Graph Wiring ────────────────────────────────────────────────────────

    void add_predecessor(TaskId pred_id) {
        predecessors_.push_back(pred_id);
        pending_deps_.fetch_add(1, std::memory_order_relaxed);
    }

    void add_successor(TaskId succ_id) {
        successors_.push_back(succ_id);
    }

    /// Called when a predecessor completes. Returns true if this task becomes ready.
    bool notify_dependency_met() {
        size_t prev = pending_deps_.fetch_sub(1, std::memory_order_acq_rel);
        return prev == 1; // Was 1, now 0 → all deps met
    }

    /// Check if task has no pending dependencies
    [[nodiscard]] bool is_ready() const noexcept {
        return pending_deps_.load(std::memory_order_acquire) == 0;
    }

    // ─── Execution ───────────────────────────────────────────────────────────

    void execute() {
        state_.store(TaskState::Running, std::memory_order_release);
        try {
            func_();
            state_.store(TaskState::Done, std::memory_order_release);
        } catch (...) {
            state_.store(TaskState::Failed, std::memory_order_release);
            exception_ = std::current_exception();
        }
    }

    void mark_ready() {
        state_.store(TaskState::Ready, std::memory_order_release);
    }

    [[nodiscard]] std::exception_ptr exception() const { return exception_; }

private:
    TaskId id_;
    TaskFunc func_;
    std::string name_;

    std::atomic<TaskState> state_;
    std::atomic<size_t> pending_deps_;

    std::vector<TaskId> predecessors_;
    std::vector<TaskId> successors_;

    std::exception_ptr exception_;
};

/// Task graph: DAG of tasks with dependency-driven execution.
///
/// Execution algorithm:
///   1. Find all tasks with 0 dependencies → submit to thread pool
///   2. When a task completes, decrement dependency count of its successors
///   3. If a successor's count reaches 0 → submit it
///   4. Repeat until all tasks are done
///
/// This is effectively Kahn's algorithm executed concurrently,
/// with the thread pool providing the parallelism.
class TaskGraph {
public:
    using TaskId = Task::TaskId;

    /// Add a task with optional dependencies.
    /// Returns task ID for use as dependency in later tasks.
    TaskId add_task(std::function<void()> func,
                    std::vector<TaskId> deps = {},
                    std::string name = "") {
        TaskId id = next_id_++;
        auto task = std::make_shared<Task>(id, std::move(func), std::move(name));

        // Wire up dependencies
        for (TaskId dep_id : deps) {
            auto it = tasks_.find(dep_id);
            if (it == tasks_.end()) {
                throw std::invalid_argument(
                    "TaskGraph: dependency " + std::to_string(dep_id) + " not found");
            }
            task->add_predecessor(dep_id);
            it->second->add_successor(id);
        }

        tasks_[id] = task;
        return id;
    }

    /// Execute all tasks on the given thread pool.
    /// Blocks until all tasks complete.
    /// Throws if any task failed.
    void execute(ThreadPool& pool) {
        if (tasks_.empty()) return;

        completed_count_.store(0, std::memory_order_relaxed);
        size_t total = tasks_.size();

        // Submit all tasks with no dependencies (roots)
        for (auto& [id, task] : tasks_) {
            if (task->is_ready()) {
                task->mark_ready();
                submit_task(pool, task);
            }
        }

        // Wait for all tasks to complete
        std::unique_lock<std::mutex> lock(done_mutex_);
        done_cv_.wait(lock, [&] {
            return completed_count_.load(std::memory_order_acquire) >= total;
        });

        // Check for failures
        for (auto& [id, task] : tasks_) {
            if (task->state() == TaskState::Failed) {
                std::rethrow_exception(task->exception());
            }
        }
    }

    /// Number of tasks in the graph
    [[nodiscard]] size_t size() const noexcept { return tasks_.size(); }

    /// Get task by ID
    [[nodiscard]] const Task& task(TaskId id) const { return *tasks_.at(id); }

    /// Reset all tasks for re-execution (resets state and dep counters)
    void reset() {
        tasks_.clear();
        next_id_ = 0;
        completed_count_.store(0, std::memory_order_relaxed);
    }

private:
    void submit_task(ThreadPool& pool, std::shared_ptr<Task> task) {
        pool.enqueue([this, task, &pool] {
            task->execute();

            // Notify successors
            for (TaskId succ_id : task->successors()) {
                auto& succ = tasks_[succ_id];
                if (succ->notify_dependency_met()) {
                    succ->mark_ready();
                    submit_task(pool, succ);
                }
            }

            // Track completion
            completed_count_.fetch_add(1, std::memory_order_release);
            done_cv_.notify_all();
        });
    }

    std::unordered_map<TaskId, std::shared_ptr<Task>> tasks_;
    TaskId next_id_ = 0;

    std::atomic<size_t> completed_count_{0};
    std::mutex done_mutex_;
    std::condition_variable done_cv_;
};

} // namespace inferx::parallel
