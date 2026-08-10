#pragma once

/// @file executor.h
/// @brief Sequential graph executor with arena-backed memory allocation.
///
/// The Executor is responsible for:
///   1. Walking the graph in topological order
///   2. Allocating memory for each node's output (from the arena)
///   3. Calling each operator's execute() with correct input/output buffers
///   4. Managing the lifecycle of intermediate tensors
///
/// Memory strategy:
///   - All intermediate tensors are allocated from a single Arena
///   - The arena is reset between inference calls (zero malloc on hot path)
///   - Input and output buffers are provided externally (not arena-managed)
///
/// This is the same execution model used by:
///   - TensorRT: enqueue() allocates from a pre-planned memory pool
///   - ONNX Runtime: SequentialExecutor with MemoryPattern planning
///   - CoreML: MILPlanExecution with pre-allocated buffer pools
///
/// Future extensions (not implemented here):
///   - Parallel executor (run independent nodes concurrently via thread pool)
///   - Streaming executor (overlap data transfer with compute)
///   - Memory planning (reuse buffers whose lifetimes don't overlap)

#include <inferx/graph/graph.h>
#include <inferx/graph/node.h>
#include <inferx/graph/operator.h>
#include <inferx/memory/arena.h>

#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <chrono>

namespace inferx::graph {

/// Result of a single execution: output tensors + profiling info.
struct ExecutionResult {
    /// Output tensor data (one per graph output, in order)
    std::vector<std::vector<float>> outputs;

    /// Per-node execution times (node_id → microseconds)
    std::unordered_map<size_t, double> node_times_us;

    /// Total execution time (microseconds)
    double total_time_us = 0.0;

    /// Peak arena usage during this execution (bytes)
    size_t peak_memory_bytes = 0;

    /// Number of nodes executed
    size_t nodes_executed = 0;
};

/// Sequential executor: runs graph nodes one at a time in topological order.
///
/// Usage:
///   Graph graph = build_my_model();
///   Executor executor(graph);
///
///   // Feed inputs
///   executor.set_input(0, input_data);
///
///   // Run inference
///   auto result = executor.execute();
///
///   // Get outputs
///   const auto& output = result.outputs[0];
class Executor {
public:
    /// Construct executor for a given graph.
    /// Validates the graph and computes execution order.
    /// @param arena_size  Size of the scratch arena (default 16 MB)
    explicit Executor(Graph& graph, size_t arena_size = 16 * 1024 * 1024)
        : graph_(graph), arena_(arena_size) {
        // Validate graph
        std::string err = graph_.validate();
        if (!err.empty()) {
            throw std::invalid_argument("Executor: invalid graph — " + err);
        }

        // Pre-compute execution order
        execution_order_ = graph_.topological_sort();

        // Pre-allocate input buffer map
        for (size_t id : graph_.input_ids()) {
            input_buffers_[id] = nullptr;
        }
    }

    // ─── Input Management ────────────────────────────────────────────────────

    /// Set input data for a graph input node.
    /// The data pointer must remain valid until execute() returns.
    /// @param input_index  Which graph input (0-based, in order of add_input calls)
    /// @param data         Pointer to input float data (must match declared shape)
    void set_input(size_t input_index, const float* data) {
        if (input_index >= graph_.input_ids().size()) {
            throw std::out_of_range(
                "Executor::set_input: index " + std::to_string(input_index) +
                " >= " + std::to_string(graph_.input_ids().size()) + " inputs");
        }
        size_t node_id = graph_.input_ids()[input_index];
        input_buffers_[node_id] = data;
    }

    /// Set input from a vector (convenience).
    void set_input(size_t input_index, const std::vector<float>& data) {
        set_input(input_index, data.data());
    }

    // ─── Execution ───────────────────────────────────────────────────────────

    /// Execute the graph: runs all operators in topological order.
    ///
    /// Algorithm:
    ///   1. Reset arena (free all intermediate memory from previous run)
    ///   2. For each node in topological order:
    ///      a. Gather input TensorDescs (from input_buffers_ or node_outputs_)
    ///      b. Allocate output buffer from arena
    ///      c. Call operator.execute(inputs, outputs)
    ///      d. Store output in node_outputs_ map
    ///   3. Copy graph outputs into result
    ///
    /// Memory lifecycle:
    ///   - Arena is reset at the START of each execute() call
    ///   - All intermediates live until the next execute() call
    ///   - Outputs are copied out (safe to use after arena reset)
    [[nodiscard]] ExecutionResult execute(bool profile = false) {
        ExecutionResult result;
        auto total_start = std::chrono::steady_clock::now();

        // Reset arena from previous execution
        arena_.reset();
        node_outputs_.clear();

        // Validate all inputs are set
        for (size_t id : graph_.input_ids()) {
            if (input_buffers_[id] == nullptr) {
                throw std::runtime_error(
                    "Executor::execute: input '" + graph_.node(id).name() + "' not set");
            }
        }

        // Execute in topological order
        for (size_t node_id : execution_order_) {
            auto& node = graph_.node(node_id);
            if (node.is_dead()) continue;

            auto node_start = std::chrono::steady_clock::now();

            // Input nodes: wrap external data as TensorDesc
            if (std::find(graph_.input_ids().begin(), graph_.input_ids().end(), node_id)
                    != graph_.input_ids().end()) {
                TensorDesc desc;
                desc.data = const_cast<float*>(input_buffers_[node_id]);
                desc.shape = node.output_shape();
                node_outputs_[node_id] = desc;
                continue;
            }

            // Gather input TensorDescs
            std::vector<TensorDesc> inputs;
            for (const auto& edge : node.inputs()) {
                auto it = node_outputs_.find(edge.producer_id);
                if (it == node_outputs_.end()) {
                    throw std::runtime_error(
                        "Executor: output of node " + std::to_string(edge.producer_id) +
                        " not available for node " + node.name());
                }
                inputs.push_back(it->second);
            }

            // Allocate output buffer from arena
            size_t out_numel = node.output_numel();
            float* out_ptr = arena_.alloc<float>(out_numel);
            if (!out_ptr) {
                throw std::runtime_error(
                    "Executor: arena out of memory for node " + node.name() +
                    " (need " + std::to_string(out_numel * sizeof(float)) + " bytes)");
            }

            TensorDesc out_desc;
            out_desc.data = out_ptr;
            out_desc.shape = node.output_shape();
            std::vector<TensorDesc> outputs = {out_desc};

            // Execute the operator
            node.op().execute(inputs, outputs);

            // Store output for downstream consumers
            node_outputs_[node_id] = out_desc;
            node.set_state(NodeState::Executed);
            ++result.nodes_executed;

            // Profile timing
            if (profile) {
                auto node_end = std::chrono::steady_clock::now();
                double us = std::chrono::duration<double, std::micro>(node_end - node_start).count();
                result.node_times_us[node_id] = us;
            }
        }

        // Collect graph outputs (copy out — safe after arena reset)
        for (size_t out_id : graph_.output_ids()) {
            auto it = node_outputs_.find(out_id);
            if (it == node_outputs_.end()) {
                throw std::runtime_error(
                    "Executor: output node " + std::to_string(out_id) + " was not executed");
            }
            const auto& desc = it->second;
            result.outputs.emplace_back(desc.data, desc.data + desc.numel());
        }

        auto total_end = std::chrono::steady_clock::now();
        result.total_time_us = std::chrono::duration<double, std::micro>(total_end - total_start).count();
        result.peak_memory_bytes = arena_.peak_usage();

        return result;
    }

    // ─── Accessors ───────────────────────────────────────────────────────────

    /// Get the execution order (topological sort result)
    [[nodiscard]] const std::vector<size_t>& execution_order() const noexcept {
        return execution_order_;
    }

    /// Get arena stats
    [[nodiscard]] size_t arena_capacity() const noexcept { return arena_.capacity(); }
    [[nodiscard]] size_t arena_peak_usage() const noexcept { return arena_.peak_usage(); }

    /// Generate execution profile report
    [[nodiscard]] std::string profile_report(const ExecutionResult& result) const {
        std::ostringstream oss;
        oss << "═══ Execution Profile ═══\n";
        oss << "Total time: " << result.total_time_us << " µs\n";
        oss << "Nodes executed: " << result.nodes_executed << "\n";
        oss << "Peak memory: " << result.peak_memory_bytes / 1024 << " KB\n";
        oss << "─────────────────────────\n";

        if (!result.node_times_us.empty()) {
            // Sort by time descending
            std::vector<std::pair<size_t, double>> sorted_times(
                result.node_times_us.begin(), result.node_times_us.end());
            std::sort(sorted_times.begin(), sorted_times.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

            for (const auto& [id, us] : sorted_times) {
                double pct = (us / result.total_time_us) * 100.0;
                oss << "  " << graph_.node(id).name()
                    << " [" << op_type_name(graph_.node(id).op_type()) << "]"
                    << ": " << us << " µs (" << pct << "%)\n";
            }
        }

        return oss.str();
    }

private:
    Graph& graph_;
    memory::Arena arena_;
    std::vector<size_t> execution_order_;

    /// External input buffers (set by user before execute)
    std::unordered_map<size_t, const float*> input_buffers_;

    /// Node output buffers (populated during execution)
    std::unordered_map<size_t, TensorDesc> node_outputs_;
};

} // namespace inferx::graph
