/// @file test_thread_pool.cpp
/// @brief Tests for thread pool, task graph, and parallel_for.

#include <gtest/gtest.h>
#include <inferx/parallel/thread_pool.h>
#include <inferx/parallel/task.h>
#include <inferx/parallel/parallel_for.h>

#include <atomic>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>

using namespace inferx::parallel;

// ═══════════════════════════════════════════════════════════════════════════════
// ThreadPool Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(ThreadPool, CreateAndDestroy) {
    ThreadPool pool(4);
    EXPECT_EQ(pool.num_threads(), 4u);
    EXPECT_FALSE(pool.is_stopped());
}

TEST(ThreadPool, SubmitAndGetResult) {
    ThreadPool pool(2);
    auto future = pool.submit([] { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST(ThreadPool, SubmitMultipleTasks) {
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 100; ++i) {
        futures.push_back(pool.submit([i] { return i * i; }));
    }
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(futures[i].get(), i * i);
    }
}

TEST(ThreadPool, Enqueue) {
    ThreadPool pool(2);
    std::atomic<int> counter{0};
    for (int i = 0; i < 50; ++i) {
        pool.enqueue([&counter] { counter.fetch_add(1); });
    }
    pool.wait_idle();
    EXPECT_EQ(counter.load(), 50);
}

TEST(ThreadPool, WaitIdle) {
    ThreadPool pool(4);
    std::atomic<int> sum{0};
    for (int i = 0; i < 1000; ++i) {
        pool.enqueue([&sum] {
            sum.fetch_add(1, std::memory_order_relaxed);
        });
    }
    pool.wait_idle();
    EXPECT_EQ(sum.load(), 1000);
}

TEST(ThreadPool, ConcurrentExecution) {
    // Verify tasks actually run in parallel (not sequential)
    ThreadPool pool(4);
    std::atomic<int> max_concurrent{0};
    std::atomic<int> current{0};

    std::vector<std::future<void>> futures;
    for (int i = 0; i < 8; ++i) {
        futures.push_back(pool.submit([&] {
            int c = current.fetch_add(1) + 1;
            // Update max concurrency seen
            int prev_max = max_concurrent.load();
            while (c > prev_max) {
                if (max_concurrent.compare_exchange_weak(prev_max, c)) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            current.fetch_sub(1);
        }));
    }
    for (auto& f : futures) f.get();

    // With 4 threads and 20ms sleep, we should see >= 2 concurrent
    EXPECT_GE(max_concurrent.load(), 2);
}

TEST(ThreadPool, Shutdown) {
    auto pool = std::make_unique<ThreadPool>(2);
    pool->enqueue([] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); });
    pool->shutdown();
    EXPECT_TRUE(pool->is_stopped());
}

TEST(ThreadPool, DefaultThreadCount) {
    ThreadPool pool; // default = hardware_concurrency
    EXPECT_GE(pool.num_threads(), 1u);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TaskGraph Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(TaskGraph, SingleTask) {
    ThreadPool pool(2);
    TaskGraph graph;
    int result = 0;
    graph.add_task([&] { result = 42; });
    graph.execute(pool);
    EXPECT_EQ(result, 42);
}

TEST(TaskGraph, IndependentTasks) {
    ThreadPool pool(4);
    TaskGraph graph;
    std::atomic<int> sum{0};

    for (int i = 0; i < 10; ++i) {
        graph.add_task([&sum, i] { sum.fetch_add(i); });
    }
    graph.execute(pool);
    EXPECT_EQ(sum.load(), 45); // 0+1+...+9 = 45
}

TEST(TaskGraph, LinearDependencyChain) {
    ThreadPool pool(4);
    TaskGraph graph;
    std::vector<int> order;
    std::mutex mtx;

    auto t0 = graph.add_task([&] {
        std::lock_guard<std::mutex> lk(mtx);
        order.push_back(0);
    });
    auto t1 = graph.add_task([&] {
        std::lock_guard<std::mutex> lk(mtx);
        order.push_back(1);
    }, {t0});
    auto t2 = graph.add_task([&] {
        std::lock_guard<std::mutex> lk(mtx);
        order.push_back(2);
    }, {t1});

    graph.execute(pool);

    // Must execute in order: 0, 1, 2
    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order[0], 0);
    EXPECT_EQ(order[1], 1);
    EXPECT_EQ(order[2], 2);
}

TEST(TaskGraph, DiamondDependency) {
    // Diamond: A → B, A → C, B+C → D
    ThreadPool pool(4);
    TaskGraph graph;
    std::atomic<int> counter{0};
    std::vector<int> exec_order;
    std::mutex mtx;

    auto a = graph.add_task([&] {
        std::lock_guard<std::mutex> lk(mtx);
        exec_order.push_back(0);
        counter.fetch_add(1);
    }, {}, "A");

    auto b = graph.add_task([&] {
        std::lock_guard<std::mutex> lk(mtx);
        exec_order.push_back(1);
        counter.fetch_add(1);
    }, {a}, "B");

    auto c = graph.add_task([&] {
        std::lock_guard<std::mutex> lk(mtx);
        exec_order.push_back(2);
        counter.fetch_add(1);
    }, {a}, "C");

    graph.add_task([&] {
        std::lock_guard<std::mutex> lk(mtx);
        exec_order.push_back(3);
        counter.fetch_add(1);
    }, {b, c}, "D");

    graph.execute(pool);

    EXPECT_EQ(counter.load(), 4);
    // A must be before B and C; B and C must be before D
    auto pos = [&](int v) {
        return std::find(exec_order.begin(), exec_order.end(), v) - exec_order.begin();
    };
    EXPECT_LT(pos(0), pos(1)); // A before B
    EXPECT_LT(pos(0), pos(2)); // A before C
    EXPECT_LT(pos(1), pos(3)); // B before D
    EXPECT_LT(pos(2), pos(3)); // C before D
}

TEST(TaskGraph, AttentionPattern) {
    // Simulates Q/K/V parallel projections → attention
    ThreadPool pool(4);
    TaskGraph graph;
    std::atomic<int> parallel_count{0};
    std::atomic<int> max_parallel{0};

    auto input = graph.add_task([&] {
        // Input ready
    }, {}, "input");

    // Q, K, V are independent (can run in parallel)
    auto q = graph.add_task([&] {
        int c = parallel_count.fetch_add(1) + 1;
        int prev = max_parallel.load();
        while (c > prev) { if (max_parallel.compare_exchange_weak(prev, c)) break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        parallel_count.fetch_sub(1);
    }, {input}, "Q");

    auto k = graph.add_task([&] {
        int c = parallel_count.fetch_add(1) + 1;
        int prev = max_parallel.load();
        while (c > prev) { if (max_parallel.compare_exchange_weak(prev, c)) break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        parallel_count.fetch_sub(1);
    }, {input}, "K");

    auto v = graph.add_task([&] {
        int c = parallel_count.fetch_add(1) + 1;
        int prev = max_parallel.load();
        while (c > prev) { if (max_parallel.compare_exchange_weak(prev, c)) break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        parallel_count.fetch_sub(1);
    }, {input}, "V");

    graph.add_task([&] {
        // Attention: needs Q, K, V
    }, {q, k, v}, "attention");

    graph.execute(pool);

    // Q, K, V should have run concurrently (max_parallel >= 2)
    EXPECT_GE(max_parallel.load(), 2);
}

// ═══════════════════════════════════════════════════════════════════════════════
// parallel_for Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(ParallelFor, BasicCorrectness) {
    ThreadPool pool(4);
    std::vector<int> data(10000, 0);

    parallel_for(pool, size_t(0), data.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            data[i] = static_cast<int>(i);
        }
    });

    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(data[i], static_cast<int>(i));
    }
}

TEST(ParallelFor, SmallRangeSequential) {
    // Below threshold, should still produce correct results (runs sequentially)
    ThreadPool pool(4);
    std::vector<int> data(100, 0);

    parallel_for(pool, size_t(0), data.size(), [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            data[i] = 1;
        }
    });

    int sum = std::accumulate(data.begin(), data.end(), 0);
    EXPECT_EQ(sum, 100);
}

TEST(ParallelFor, ForEach) {
    ThreadPool pool(4);
    std::vector<float> data(5000);
    std::iota(data.begin(), data.end(), 0.0f);

    // Square each element in parallel
    parallel_for_each(pool, size_t(0), data.size(), [&](size_t i) {
        data[i] = data[i] * data[i];
    });

    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 1.0f);
    EXPECT_FLOAT_EQ(data[100], 10000.0f);
}

TEST(ParallelFor, Reduce) {
    ThreadPool pool(4);
    std::vector<float> data(10000);
    std::iota(data.begin(), data.end(), 1.0f); // 1, 2, ..., 10000

    float sum = parallel_reduce(pool, size_t(0), data.size(), 0.0f,
        [&](size_t i) { return data[i]; },
        [](float a, float b) { return a + b; });

    // Sum of 1..10000 = 10000 * 10001 / 2 = 50005000
    // Note: float32 has limited precision for large sums, use NEAR
    EXPECT_NEAR(sum, 50005000.0f, 1024.0f);
}

TEST(ParallelFor, ReduceMax) {
    ThreadPool pool(4);
    std::vector<float> data(10000);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(static_cast<float>(i) * 0.01f);
    }

    float max_val = parallel_reduce(pool, size_t(0), data.size(),
        -std::numeric_limits<float>::infinity(),
        [&](size_t i) { return data[i]; },
        [](float a, float b) { return std::max(a, b); });

    float seq_max = *std::max_element(data.begin(), data.end());
    EXPECT_FLOAT_EQ(max_val, seq_max);
}

TEST(ParallelFor, EmptyRange) {
    ThreadPool pool(4);
    int called = 0;
    parallel_for(pool, size_t(5), size_t(5), [&](size_t, size_t) {
        called++;
    });
    EXPECT_EQ(called, 0);
}
