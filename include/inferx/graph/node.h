#pragma once

/// @file node.h
/// @brief Graph node — wraps an operator with input/output edges and metadata.
///
/// A computational graph is a DAG (directed acyclic graph) where:
///   - Nodes represent operations (MatMul, ReLU, Add, etc.)
///   - Edges represent data flow (tensors moving between operations)
///
/// Each node has:
///   - An operator (defines the computation)
///   - Input edges (indices of producer nodes)
///   - Output shape (computed via shape inference)
///   - Execution state (pending, ready, executed)
///   - A unique ID for graph manipulation
///
/// The node does NOT own tensor memory — that's the executor's job.
/// This separation allows the optimizer to rewrite the graph without
/// worrying about memory management.
///
/// Design parallels:
///   - TensorRT ILayer: each layer has inputs/outputs, type, name
///   - ONNX NodeProto: op_type, inputs[], outputs[], attributes
///   - TensorFlow NodeDef: op, input[], device, attr{}

#include <inferx/graph/operator.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace inferx::graph {

/// Execution state of a node (used by the executor)
enum class NodeState {
    Pending,    ///< Not yet ready (waiting for inputs)
    Ready,      ///< All inputs available, can be executed
    Executed,   ///< Execution complete, output available
    Dead        ///< Removed by optimizer (dead code elimination)
};

/// Edge: represents a data dependency between two nodes.
/// An edge connects one output of a producer node to one input of a consumer node.
struct Edge {
    size_t producer_id;     ///< Node that produces this tensor
    size_t output_index;    ///< Which output of the producer (usually 0)

    bool operator==(const Edge& other) const noexcept {
        return producer_id == other.producer_id && output_index == other.output_index;
    }
};

/// Graph Node: combines an operator with its connectivity and metadata.
class Node {
public:
    /// Construct a node with given ID and operator.
    Node(size_t id, std::shared_ptr<Operator> op, std::string name = "")
        : id_(id), op_(std::move(op)),
          name_(name.empty() ? op_->name() + "_" + std::to_string(id) : std::move(name)),
          state_(NodeState::Pending) {}

    // ─── Accessors ───────────────────────────────────────────────────────────

    /// Unique node ID within the graph
    [[nodiscard]] size_t id() const noexcept { return id_; }

    /// Human-readable name (e.g., "MatMul_0", "ReLU_1")
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    /// The operator this node executes
    [[nodiscard]] const Operator& op() const noexcept { return *op_; }
    [[nodiscard]] std::shared_ptr<Operator> op_ptr() const noexcept { return op_; }

    /// Operator type (shortcut for op().type())
    [[nodiscard]] OpType op_type() const noexcept { return op_->type(); }

    /// Current execution state
    [[nodiscard]] NodeState state() const noexcept { return state_; }
    void set_state(NodeState s) noexcept { state_ = s; }

    // ─── Connectivity ────────────────────────────────────────────────────────

    /// Input edges: which nodes produce this node's inputs
    [[nodiscard]] const std::vector<Edge>& inputs() const noexcept { return inputs_; }

    /// Add an input edge (connects producer → this node)
    void add_input(Edge edge) { inputs_.push_back(edge); }

    /// Set all input edges at once (used by optimizer when rewriting)
    void set_inputs(std::vector<Edge> edges) { inputs_ = std::move(edges); }

    /// Consumer nodes: which nodes read this node's output
    [[nodiscard]] const std::vector<size_t>& consumers() const noexcept { return consumers_; }

    /// Register a consumer (called during graph construction)
    void add_consumer(size_t consumer_id) { consumers_.push_back(consumer_id); }

    /// Remove a consumer (called during graph optimization)
    void remove_consumer(size_t consumer_id) {
        consumers_.erase(
            std::remove(consumers_.begin(), consumers_.end(), consumer_id),
            consumers_.end());
    }

    /// Number of consumers (0 = graph output or dead node)
    [[nodiscard]] size_t num_consumers() const noexcept { return consumers_.size(); }

    // ─── Shape Information ───────────────────────────────────────────────────

    /// Output shape (set during shape inference pass)
    [[nodiscard]] const std::vector<size_t>& output_shape() const noexcept { return output_shape_; }
    void set_output_shape(std::vector<size_t> shape) { output_shape_ = std::move(shape); }

    /// Number of elements in output
    [[nodiscard]] size_t output_numel() const noexcept {
        if (output_shape_.empty()) return 0;
        size_t n = 1;
        for (auto d : output_shape_) n *= d;
        return n;
    }

    /// Output size in bytes
    [[nodiscard]] size_t output_size_bytes() const noexcept {
        return output_numel() * sizeof(float);
    }

    // ─── Graph Metadata ──────────────────────────────────────────────────────

    /// Whether this node is a graph input (no incoming edges from other ops)
    [[nodiscard]] bool is_input() const noexcept { return inputs_.empty(); }

    /// Whether this node is a graph output (no consumers)
    [[nodiscard]] bool is_output() const noexcept { return consumers_.empty(); }

    /// Whether this node was marked dead by the optimizer
    [[nodiscard]] bool is_dead() const noexcept { return state_ == NodeState::Dead; }

    /// Mark this node as dead (optimizer removes it)
    void mark_dead() noexcept { state_ = NodeState::Dead; }

    // ─── Operator Replacement (for fusion) ───────────────────────────────────

    /// Replace the operator (used by fusion pass: MatMul+ReLU → FusedMatMulReLU)
    void replace_operator(std::shared_ptr<Operator> new_op) {
        op_ = std::move(new_op);
    }

    /// String representation for debugging
    [[nodiscard]] std::string to_string() const {
        std::string s = name_ + " [" + op_type_name(op_type()) + "]";
        s += " inputs={";
        for (size_t i = 0; i < inputs_.size(); ++i) {
            if (i > 0) s += ", ";
            s += std::to_string(inputs_[i].producer_id);
        }
        s += "} consumers={";
        for (size_t i = 0; i < consumers_.size(); ++i) {
            if (i > 0) s += ", ";
            s += std::to_string(consumers_[i]);
        }
        s += "}";
        if (!output_shape_.empty()) {
            s += " shape=[";
            for (size_t i = 0; i < output_shape_.size(); ++i) {
                if (i > 0) s += "×";
                s += std::to_string(output_shape_[i]);
            }
            s += "]";
        }
        return s;
    }

private:
    size_t id_;
    std::shared_ptr<Operator> op_;
    std::string name_;
    NodeState state_;

    std::vector<Edge> inputs_;          ///< Where this node gets its data
    std::vector<size_t> consumers_;     ///< Who reads this node's output
    std::vector<size_t> output_shape_;  ///< Inferred output shape
};

} // namespace inferx::graph
