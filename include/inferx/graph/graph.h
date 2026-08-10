#pragma once

/// @file graph.h
/// @brief Directed Acyclic Graph (DAG) for computational graphs.
///
/// The Graph is the central data structure of any AI compiler/runtime.
/// It represents the complete computation as a set of nodes connected by edges:
///
///   Input → MatMul → ReLU → MatMul → Softmax → Output
///                ↗              ↗
///           Weights_1      Weights_2
///
/// Key operations:
///   1. Construction: add_node(), connect()
///   2. Validation: validate() — checks shapes, no cycles, connectivity
///   3. Topological sort: topo_sort() — determines execution order
///   4. Shape inference: infer_shapes() — propagates shapes through the graph
///   5. Visualization: to_string() — human-readable graph dump
///
/// The graph is immutable during execution. All modifications (fusion, pruning)
/// happen before execution via the optimizer.
///
/// Design parallels:
///   - TensorRT INetworkDefinition: addInput(), addLayer(), markOutput()
///   - ONNX GraphProto: node[], input[], output[], initializer[]
///   - TensorFlow GraphDef: node[] with implicit edges via input names
///   - XLA HloComputation: instructions() in topological order

#include <inferx/graph/node.h>
#include <inferx/graph/operator.h>

#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace inferx::graph {

/// Graph construction and analysis.
class Graph {
public:
    Graph() = default;

    // ─── Construction ────────────────────────────────────────────────────────

    /// Add an input node (external data fed into the graph).
    /// Input nodes have no operator — they just hold a shape.
    /// Returns the node ID.
    size_t add_input(const std::string& name, std::vector<size_t> shape) {
        size_t id = next_id_++;
        auto op = std::make_shared<IdentityOp>();
        auto node = std::make_unique<Node>(id, op, name);
        node->set_output_shape(std::move(shape));
        node->set_state(NodeState::Ready); // Inputs are always ready

        input_ids_.push_back(id);
        nodes_[id] = std::move(node);
        return id;
    }

    /// Add an operator node with specified inputs.
    /// Automatically infers output shape and wires up connectivity.
    /// Returns the node ID.
    size_t add_node(std::shared_ptr<Operator> op,
                    const std::vector<size_t>& input_ids,
                    const std::string& name = "") {
        // Validate inputs exist
        for (auto in_id : input_ids) {
            if (nodes_.find(in_id) == nodes_.end()) {
                throw std::invalid_argument(
                    "Graph::add_node: input node " + std::to_string(in_id) + " does not exist");
            }
        }

        // Validate input count matches operator expectation
        if (input_ids.size() != op->num_inputs()) {
            throw std::invalid_argument(
                "Graph::add_node: operator " + op->name() + " expects " +
                std::to_string(op->num_inputs()) + " inputs, got " +
                std::to_string(input_ids.size()));
        }

        size_t id = next_id_++;
        auto node = std::make_unique<Node>(id, op, name);

        // Wire up input edges
        for (auto in_id : input_ids) {
            node->add_input(Edge{in_id, 0});
            nodes_[in_id]->add_consumer(id);
        }

        // Infer output shape
        std::vector<std::vector<size_t>> input_shapes;
        for (auto in_id : input_ids) {
            input_shapes.push_back(nodes_[in_id]->output_shape());
        }
        auto out_shape = op->infer_output_shape(input_shapes);
        node->set_output_shape(std::move(out_shape));

        nodes_[id] = std::move(node);
        return id;
    }

    /// Mark a node as a graph output.
    /// Graph outputs are the final results that the executor must produce.
    void mark_output(size_t node_id) {
        if (nodes_.find(node_id) == nodes_.end()) {
            throw std::invalid_argument(
                "Graph::mark_output: node " + std::to_string(node_id) + " does not exist");
        }
        output_ids_.push_back(node_id);
    }

    // ─── Accessors ───────────────────────────────────────────────────────────

    /// Get a node by ID
    [[nodiscard]] Node& node(size_t id) { return *nodes_.at(id); }
    [[nodiscard]] const Node& node(size_t id) const { return *nodes_.at(id); }

    /// Get node pointer (for optimizer manipulation)
    [[nodiscard]] Node* node_ptr(size_t id) { return nodes_.at(id).get(); }

    /// All node IDs in insertion order
    [[nodiscard]] std::vector<size_t> node_ids() const {
        std::vector<size_t> ids;
        ids.reserve(nodes_.size());
        for (const auto& [id, _] : nodes_) {
            ids.push_back(id);
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    /// Input node IDs
    [[nodiscard]] const std::vector<size_t>& input_ids() const noexcept { return input_ids_; }

    /// Output node IDs
    [[nodiscard]] const std::vector<size_t>& output_ids() const noexcept { return output_ids_; }

    /// Total number of nodes (including inputs)
    [[nodiscard]] size_t size() const noexcept { return nodes_.size(); }

    /// Check if a node exists
    [[nodiscard]] bool has_node(size_t id) const { return nodes_.find(id) != nodes_.end(); }

    // ─── Topological Sort ────────────────────────────────────────────────────

    /// Compute topological order using Kahn's algorithm (BFS-based).
    ///
    /// Kahn's algorithm:
    ///   1. Compute in-degree of each node
    ///   2. Enqueue all nodes with in-degree 0 (inputs)
    ///   3. Process queue: for each node, decrement in-degree of consumers
    ///   4. When a consumer's in-degree reaches 0, enqueue it
    ///   5. If all nodes processed → valid DAG. Otherwise → cycle detected.
    ///
    /// Returns: node IDs in valid execution order.
    /// Throws: if the graph contains a cycle.
    [[nodiscard]] std::vector<size_t> topological_sort() const {
        // Compute in-degrees (number of unprocessed inputs per node)
        std::unordered_map<size_t, size_t> in_degree;
        for (const auto& [id, node] : nodes_) {
            if (node->is_dead()) continue;
            in_degree[id] = 0;
        }
        for (const auto& [id, node] : nodes_) {
            if (node->is_dead()) continue;
            for (const auto& edge : node->inputs()) {
                // Each input edge contributes 1 to this node's in-degree
                // (unless the producer is dead)
                if (!nodes_.at(edge.producer_id)->is_dead()) {
                    in_degree[id]++;
                }
            }
        }

        // Enqueue nodes with in-degree 0 (graph inputs)
        std::queue<size_t> ready;
        for (const auto& [id, deg] : in_degree) {
            if (deg == 0) ready.push(id);
        }

        // BFS
        std::vector<size_t> order;
        order.reserve(nodes_.size());

        while (!ready.empty()) {
            size_t id = ready.front();
            ready.pop();
            order.push_back(id);

            // Decrement in-degree of all consumers
            for (size_t consumer_id : nodes_.at(id)->consumers()) {
                if (nodes_.at(consumer_id)->is_dead()) continue;
                if (--in_degree[consumer_id] == 0) {
                    ready.push(consumer_id);
                }
            }
        }

        // Check for cycles
        size_t live_nodes = 0;
        for (const auto& [id, node] : nodes_) {
            if (!node->is_dead()) ++live_nodes;
        }
        if (order.size() != live_nodes) {
            throw std::runtime_error(
                "Graph::topological_sort: cycle detected! Processed " +
                std::to_string(order.size()) + " of " + std::to_string(live_nodes) + " nodes");
        }

        return order;
    }

    // ─── Validation ──────────────────────────────────────────────────────────

    /// Validate the graph structure.
    /// Checks:
    ///   1. All referenced nodes exist
    ///   2. No cycles (topological sort succeeds)
    ///   3. All operators have correct input count
    ///   4. At least one input and one output
    /// Returns empty string on success, error message on failure.
    [[nodiscard]] std::string validate() const {
        if (input_ids_.empty()) {
            return "Graph has no inputs";
        }
        if (output_ids_.empty()) {
            return "Graph has no outputs";
        }

        // Check all output nodes exist
        for (auto id : output_ids_) {
            if (nodes_.find(id) == nodes_.end()) {
                return "Output node " + std::to_string(id) + " does not exist";
            }
        }

        // Check connectivity: all input edges reference valid nodes
        for (const auto& [id, node] : nodes_) {
            if (node->is_dead()) continue;
            for (const auto& edge : node->inputs()) {
                if (nodes_.find(edge.producer_id) == nodes_.end()) {
                    return "Node " + node->name() + " references non-existent input " +
                           std::to_string(edge.producer_id);
                }
            }
        }

        // Check for cycles via topological sort
        try {
            topological_sort();
        } catch (const std::runtime_error& e) {
            return e.what();
        }

        return ""; // Valid
    }

    // ─── Shape Inference ─────────────────────────────────────────────────────

    /// Re-run shape inference on all nodes in topological order.
    /// Useful after graph rewrites (fusion may change shapes).
    void infer_shapes() {
        auto order = topological_sort();
        for (size_t id : order) {
            auto& n = *nodes_[id];
            if (n.is_dead()) continue;
            if (n.inputs().empty()) continue; // Input nodes already have shapes

            std::vector<std::vector<size_t>> input_shapes;
            for (const auto& edge : n.inputs()) {
                input_shapes.push_back(nodes_[edge.producer_id]->output_shape());
            }
            auto out_shape = n.op().infer_output_shape(input_shapes);
            n.set_output_shape(std::move(out_shape));
        }
    }

    // ─── Graph Manipulation (for optimizer) ──────────────────────────────────

    /// Remove a node from the graph (mark as dead + update connectivity).
    /// Does NOT delete the node — just marks it dead so topo_sort skips it.
    void remove_node(size_t id) {
        auto& node = *nodes_.at(id);
        node.mark_dead();

        // Remove from consumers of its inputs
        for (const auto& edge : node.inputs()) {
            if (has_node(edge.producer_id)) {
                nodes_[edge.producer_id]->remove_consumer(id);
            }
        }
    }

    /// Replace a node's operator (for fusion).
    /// Also updates the node's input edges.
    void replace_node_op(size_t id, std::shared_ptr<Operator> new_op,
                         std::vector<Edge> new_inputs) {
        auto& n = *nodes_[id];

        // Remove old consumer registrations
        for (const auto& edge : n.inputs()) {
            if (has_node(edge.producer_id)) {
                nodes_[edge.producer_id]->remove_consumer(id);
            }
        }

        // Set new operator and inputs
        n.replace_operator(std::move(new_op));
        n.set_inputs(std::move(new_inputs));

        // Register new consumer relationships
        for (const auto& edge : n.inputs()) {
            if (has_node(edge.producer_id)) {
                nodes_[edge.producer_id]->add_consumer(id);
            }
        }
    }

    // ─── Statistics ──────────────────────────────────────────────────────────

    /// Count of live (non-dead) nodes
    [[nodiscard]] size_t live_node_count() const noexcept {
        size_t count = 0;
        for (const auto& [id, node] : nodes_) {
            if (!node->is_dead()) ++count;
        }
        return count;
    }

    /// Total memory required for all intermediate tensors (in bytes)
    [[nodiscard]] size_t total_intermediate_bytes() const noexcept {
        size_t total = 0;
        for (const auto& [id, node] : nodes_) {
            if (node->is_dead()) continue;
            if (std::find(input_ids_.begin(), input_ids_.end(), id) != input_ids_.end()) {
                continue; // Skip inputs (externally provided)
            }
            total += node->output_size_bytes();
        }
        return total;
    }

    // ─── Visualization ───────────────────────────────────────────────────────

    /// Human-readable graph dump (for debugging)
    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "Graph: " << live_node_count() << " nodes ("
            << input_ids_.size() << " inputs, "
            << output_ids_.size() << " outputs)\n";
        oss << "─────────────────────────────────────────\n";

        try {
            auto order = topological_sort();
            for (size_t id : order) {
                const auto& n = *nodes_.at(id);
                oss << "  " << n.to_string() << "\n";
            }
        } catch (...) {
            // If topo sort fails, just dump all nodes
            for (const auto& [id, node] : nodes_) {
                if (!node->is_dead()) {
                    oss << "  " << node->to_string() << "\n";
                }
            }
        }

        oss << "─────────────────────────────────────────\n";
        oss << "Intermediate memory: "
            << total_intermediate_bytes() / 1024 << " KB\n";
        return oss.str();
    }

private:
    std::unordered_map<size_t, std::unique_ptr<Node>> nodes_;
    std::vector<size_t> input_ids_;
    std::vector<size_t> output_ids_;
    size_t next_id_ = 0;
};

} // namespace inferx::graph
