#pragma once

/// @file optimizer.h
/// @brief Graph optimization passes: dead node elimination, operator fusion.
///
/// Graph optimization is the core of AI compilers. The idea is simple:
/// transform the graph into an equivalent but faster/smaller version
/// BEFORE execution, so the runtime pays zero cost for the optimization.
///
/// Passes implemented here:
///
/// 1. Dead Node Elimination (DNE)
///    Remove nodes whose output is never consumed by any downstream node.
///    Equivalent to dead code elimination in traditional compilers.
///    Example: a debug-only node that was left in the graph.
///
/// 2. Identity Elimination
///    Remove Identity (pass-through) nodes by rewiring consumers directly
///    to the identity's input. Identities appear after graph rewrites.
///
/// 3. Operator Fusion: MatMul + Activation → FusedMatMulActivation
///    The most impactful optimization for inference. Eliminates:
///      - One intermediate tensor allocation (M×N floats)
///      - One full memory pass over M×N elements
///    For a 1024×1024 matmul, this saves 4 MB of memory traffic.
///
///    Pattern: MatMul → ReLU (single consumer)
///    Rewrite: FusedMatMulReLU (combines both into one node)
///
///    Why this works: the activation is applied immediately after matmul
///    completes each output row, while the data is still in CPU registers/L1.
///    Without fusion, the matmul writes to memory, then ReLU reads it back.
///
/// Real-world fusion examples:
///   - TensorRT: Conv + BatchNorm + ReLU → single fused kernel
///   - oneDNN: MatMul + BiasAdd + Eltwise → matmul_post_ops
///   - XLA: dot + add + maximum → fused computation
///   - cuDNN: ConvBiasActivation (single API call for 3 ops)
///
/// Pass ordering matters:
///   1. Dead Node Elimination (clean up first)
///   2. Identity Elimination (simplify graph structure)
///   3. Operator Fusion (exploit simplified structure)
///   4. Shape re-inference (update after rewrites)

#include <inferx/graph/graph.h>
#include <inferx/graph/node.h>
#include <inferx/graph/operator.h>

#include <cstddef>
#include <string>
#include <vector>
#include <algorithm>

namespace inferx::graph {

/// Statistics reported by optimization passes.
struct OptimizationStats {
    size_t dead_nodes_removed = 0;
    size_t identities_removed = 0;
    size_t fusions_applied = 0;
    size_t nodes_before = 0;
    size_t nodes_after = 0;
    size_t memory_saved_bytes = 0;  ///< Intermediate memory eliminated by fusion

    [[nodiscard]] std::string summary() const {
        std::string s = "Optimization Results:\n";
        s += "  Nodes: " + std::to_string(nodes_before) + " → " + std::to_string(nodes_after) + "\n";
        s += "  Dead nodes removed: " + std::to_string(dead_nodes_removed) + "\n";
        s += "  Identities removed: " + std::to_string(identities_removed) + "\n";
        s += "  Fusions applied: " + std::to_string(fusions_applied) + "\n";
        if (memory_saved_bytes > 0) {
            s += "  Memory saved: " + std::to_string(memory_saved_bytes / 1024) + " KB\n";
        }
        return s;
    }
};

// ─── Individual Optimization Passes ──────────────────────────────────────────

/// Dead Node Elimination: remove nodes whose output is never used.
///
/// A node is "dead" if:
///   - It has zero consumers AND
///   - It is NOT a graph output
///
/// This runs iteratively: removing a dead node may make its inputs dead too.
/// (Same as liveness analysis in compiler IRs.)
inline size_t eliminate_dead_nodes(Graph& graph) {
    size_t removed = 0;
    bool changed = true;

    while (changed) {
        changed = false;
        for (size_t id : graph.node_ids()) {
            if (!graph.has_node(id)) continue;
            auto& node = graph.node(id);
            if (node.is_dead()) continue;

            // Skip graph inputs and outputs
            const auto& inputs = graph.input_ids();
            const auto& outputs = graph.output_ids();
            if (std::find(inputs.begin(), inputs.end(), id) != inputs.end()) continue;
            if (std::find(outputs.begin(), outputs.end(), id) != outputs.end()) continue;

            // If no consumers → dead
            if (node.num_consumers() == 0) {
                graph.remove_node(id);
                ++removed;
                changed = true;
            }
        }
    }

    return removed;
}

/// Identity Elimination: remove pass-through nodes.
///
/// Pattern: A → Identity → B
/// Rewrite: A → B (rewire B's input to point at A directly)
///
/// This commonly appears after other graph transformations.
inline size_t eliminate_identities(Graph& graph) {
    size_t removed = 0;

    for (size_t id : graph.node_ids()) {
        if (!graph.has_node(id)) continue;
        auto& node = graph.node(id);
        if (node.is_dead()) continue;
        if (node.op_type() != OpType::Identity) continue;

        // Skip graph inputs (they use Identity as placeholder)
        const auto& inputs = graph.input_ids();
        if (std::find(inputs.begin(), inputs.end(), id) != inputs.end()) continue;

        // This identity has exactly one input
        if (node.inputs().empty()) continue;
        size_t source_id = node.inputs()[0].producer_id;

        // Rewire all consumers to point at the identity's input instead
        for (size_t consumer_id : node.consumers()) {
            auto* consumer = graph.node_ptr(consumer_id);
            auto consumer_inputs = consumer->inputs();
            for (auto& edge : consumer_inputs) {
                if (edge.producer_id == id) {
                    edge.producer_id = source_id;
                }
            }
            consumer->set_inputs(consumer_inputs);

            // Register consumer with the source node
            graph.node(source_id).add_consumer(consumer_id);
        }

        // Remove the identity node
        graph.remove_node(id);
        ++removed;
    }

    return removed;
}

/// Operator Fusion: MatMul + Activation → Fused operator.
///
/// Pattern matching:
///   1. Find a MatMul node
///   2. Check if it has exactly ONE consumer
///   3. Check if that consumer is ReLU or GELU
///   4. Check if the activation has exactly one input (the matmul)
///   5. If all conditions met: replace activation node with FusedMatMulReLU/GELU
///      and rewire its inputs to be the matmul's inputs
///
/// Why "single consumer" is required:
///   If the matmul output is used by multiple downstream nodes, we can't
///   eliminate the intermediate buffer — other consumers still need it.
///   This is the same constraint TensorRT uses for fusion eligibility.
inline size_t fuse_matmul_activation(Graph& graph) {
    size_t fused = 0;

    for (size_t id : graph.node_ids()) {
        if (!graph.has_node(id)) continue;
        auto& matmul_node = graph.node(id);
        if (matmul_node.is_dead()) continue;
        if (matmul_node.op_type() != OpType::MatMul) continue;

        // MatMul must have exactly one consumer
        if (matmul_node.num_consumers() != 1) continue;

        size_t consumer_id = matmul_node.consumers()[0];
        auto& activation_node = graph.node(consumer_id);
        if (activation_node.is_dead()) continue;

        // Consumer must be an activation we can fuse
        std::shared_ptr<Operator> fused_op;
        if (activation_node.op_type() == OpType::ReLU) {
            fused_op = std::make_shared<FusedMatMulReLUOp>();
        } else if (activation_node.op_type() == OpType::GELU) {
            fused_op = std::make_shared<FusedMatMulGELUOp>();
        } else {
            continue; // Not a fusable activation
        }

        // Activation must have exactly one input (the matmul)
        if (activation_node.inputs().size() != 1) continue;
        if (activation_node.inputs()[0].producer_id != id) continue;

        // ─── Apply Fusion ─────────────────────────────────────────────────────
        // Replace activation node's operator with the fused one,
        // and rewire its inputs to be the matmul's original inputs.

        // Get matmul's inputs (the two matrices A and B)
        auto matmul_inputs = matmul_node.inputs();

        // Replace the activation node with fused op + matmul's inputs
        graph.replace_node_op(consumer_id, fused_op, matmul_inputs);

        // The matmul node is now dead (its only consumer was rewired)
        graph.remove_node(id);

        ++fused;
    }

    // Re-infer shapes after fusion
    if (fused > 0) {
        graph.infer_shapes();
    }

    return fused;
}

// ─── Combined Optimizer ──────────────────────────────────────────────────────

/// Run all optimization passes in the correct order.
/// Returns statistics about what was optimized.
inline OptimizationStats optimize(Graph& graph) {
    OptimizationStats stats;
    stats.nodes_before = graph.live_node_count();

    size_t memory_before = graph.total_intermediate_bytes();

    // Pass 1: Dead node elimination
    stats.dead_nodes_removed = eliminate_dead_nodes(graph);

    // Pass 2: Identity elimination
    stats.identities_removed = eliminate_identities(graph);

    // Pass 3: Operator fusion (MatMul + Activation)
    stats.fusions_applied = fuse_matmul_activation(graph);

    // Pass 4: Dead node elimination again (fusion may create dead nodes)
    stats.dead_nodes_removed += eliminate_dead_nodes(graph);

    stats.nodes_after = graph.live_node_count();

    size_t memory_after = graph.total_intermediate_bytes();
    stats.memory_saved_bytes = (memory_before > memory_after)
        ? memory_before - memory_after : 0;

    return stats;
}

} // namespace inferx::graph
