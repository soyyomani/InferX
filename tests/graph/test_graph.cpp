/// @file test_graph.cpp
/// @brief Correctness tests for computational graph, executor, and optimizer.

#include <gtest/gtest.h>
#include <inferx/graph/graph.h>
#include <inferx/graph/executor.h>
#include <inferx/graph/optimizer.h>

#include <vector>
#include <cmath>
#include <numeric>

using namespace inferx::graph;

// ═══════════════════════════════════════════════════════════════════════════════
// Operator Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Operators, ReLU) {
    ReLUOp op;
    std::vector<float> input = {-2, -1, 0, 1, 2};
    std::vector<float> output(5);
    TensorDesc in{input.data(), {5}};
    TensorDesc out{output.data(), {5}};
    std::vector<TensorDesc> ins = {in};
    std::vector<TensorDesc> outs = {out};
    op.execute(ins, outs);
    EXPECT_EQ(output, (std::vector<float>{0, 0, 0, 1, 2}));
}

TEST(Operators, GELU) {
    GELUOp op;
    std::vector<float> input = {0.0f, 1.0f, -1.0f};
    std::vector<float> output(3);
    TensorDesc in{input.data(), {3}};
    TensorDesc out{output.data(), {3}};
    std::vector<TensorDesc> ins = {in};
    std::vector<TensorDesc> outs = {out};
    op.execute(ins, outs);
    EXPECT_NEAR(output[0], 0.0f, 1e-5f);      // GELU(0) = 0
    EXPECT_NEAR(output[1], 0.8412f, 1e-3f);   // GELU(1) ≈ 0.841
    EXPECT_NEAR(output[2], -0.1588f, 1e-3f);  // GELU(-1) ≈ -0.159
}

TEST(Operators, Add) {
    AddOp op;
    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {5, 6, 7, 8};
    std::vector<float> c(4);
    TensorDesc ta{a.data(), {4}};
    TensorDesc tb{b.data(), {4}};
    TensorDesc tc{c.data(), {4}};
    std::vector<TensorDesc> ins = {ta, tb};
    std::vector<TensorDesc> outs = {tc};
    op.execute(ins, outs);
    EXPECT_EQ(c, (std::vector<float>{6, 8, 10, 12}));
}

TEST(Operators, MatMul2x2) {
    MatMulOp op;
    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {5, 6, 7, 8};
    std::vector<float> C(4);
    TensorDesc ta{A.data(), {2, 2}};
    TensorDesc tb{B.data(), {2, 2}};
    TensorDesc tc{C.data(), {2, 2}};
    std::vector<TensorDesc> ins = {ta, tb};
    std::vector<TensorDesc> outs = {tc};
    op.execute(ins, outs);
    EXPECT_EQ(C, (std::vector<float>{19, 22, 43, 50}));
}

TEST(Operators, Softmax) {
    SoftmaxOp op;
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3);
    TensorDesc in{input.data(), {3}};
    TensorDesc out{output.data(), {3}};
    std::vector<TensorDesc> ins = {in};
    std::vector<TensorDesc> outs = {out};
    op.execute(ins, outs);
    // Sum should be 1.0
    float sum = output[0] + output[1] + output[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
    // Monotonic
    EXPECT_LT(output[0], output[1]);
    EXPECT_LT(output[1], output[2]);
}

TEST(Operators, ShapeInference) {
    MatMulOp op;
    auto shape = op.infer_output_shape({{4, 8}, {8, 16}});
    EXPECT_EQ(shape, (std::vector<size_t>{4, 16}));
}

TEST(Operators, ShapeInferenceMismatch) {
    MatMulOp op;
    EXPECT_THROW(op.infer_output_shape({{4, 8}, {9, 16}}), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Graph Construction Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Graph, BasicConstruction) {
    Graph g;
    auto in = g.add_input("input", {4, 8});
    auto w = g.add_input("weights", {8, 16});
    auto mm = g.add_node(std::make_shared<MatMulOp>(), {in, w});
    auto relu = g.add_node(std::make_shared<ReLUOp>(), {mm});
    g.mark_output(relu);

    EXPECT_EQ(g.size(), 4u);
    EXPECT_EQ(g.input_ids().size(), 2u);
    EXPECT_EQ(g.output_ids().size(), 1u);
    EXPECT_EQ(g.node(relu).output_shape(), (std::vector<size_t>{4, 16}));
}

TEST(Graph, Validation) {
    Graph g;
    auto in = g.add_input("x", {4});
    auto relu = g.add_node(std::make_shared<ReLUOp>(), {in});
    g.mark_output(relu);
    EXPECT_EQ(g.validate(), "");
}

TEST(Graph, ValidationNoInputs) {
    Graph g;
    EXPECT_NE(g.validate(), "");
}

TEST(Graph, TopologicalSort) {
    Graph g;
    auto in = g.add_input("x", {4});
    auto relu1 = g.add_node(std::make_shared<ReLUOp>(), {in});
    auto relu2 = g.add_node(std::make_shared<ReLUOp>(), {relu1});
    g.mark_output(relu2);

    auto order = g.topological_sort();
    ASSERT_EQ(order.size(), 3u);
    // Input must come before relu1, relu1 before relu2
    auto pos_in = std::find(order.begin(), order.end(), in);
    auto pos_r1 = std::find(order.begin(), order.end(), relu1);
    auto pos_r2 = std::find(order.begin(), order.end(), relu2);
    EXPECT_LT(pos_in, pos_r1);
    EXPECT_LT(pos_r1, pos_r2);
}

TEST(Graph, InvalidInputCount) {
    Graph g;
    auto in = g.add_input("x", {4});
    // ReLU expects 1 input, giving 0 should fail
    EXPECT_THROW(g.add_node(std::make_shared<ReLUOp>(), {}), std::invalid_argument);
}

TEST(Graph, ToString) {
    Graph g;
    auto in = g.add_input("x", {2, 3});
    auto relu = g.add_node(std::make_shared<ReLUOp>(), {in});
    g.mark_output(relu);
    std::string s = g.to_string();
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("ReLU"), std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Executor Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Executor, SimpleReLU) {
    Graph g;
    auto in = g.add_input("x", {4});
    auto relu = g.add_node(std::make_shared<ReLUOp>(), {in});
    g.mark_output(relu);

    Executor exec(g);
    std::vector<float> input = {-1, 0, 1, 2};
    exec.set_input(0, input);
    auto result = exec.execute();

    ASSERT_EQ(result.outputs.size(), 1u);
    EXPECT_EQ(result.outputs[0], (std::vector<float>{0, 0, 1, 2}));
}

TEST(Executor, MatMulReLUPipeline) {
    // Build: Input(2×3) × Weights(3×2) → ReLU → Output(2×2)
    Graph g;
    auto in = g.add_input("input", {2, 3});
    auto w = g.add_input("weights", {3, 2});
    auto mm = g.add_node(std::make_shared<MatMulOp>(), {in, w});
    auto relu = g.add_node(std::make_shared<ReLUOp>(), {mm});
    g.mark_output(relu);

    Executor exec(g);
    std::vector<float> input = {1, 2, 3, 4, 5, 6};   // 2×3
    std::vector<float> weights = {1, -1, -1, 1, 1, -1}; // 3×2
    exec.set_input(0, input);
    exec.set_input(1, weights);
    auto result = exec.execute();

    // MatMul: [1,2,3]×[[1,-1],[-1,1],[1,-1]] = [1*1+2*(-1)+3*1, 1*(-1)+2*1+3*(-1)]
    //         = [2, -2]
    //         [4*1+5*(-1)+6*1, 4*(-1)+5*1+6*(-1)] = [5, -5]
    // ReLU: [2, 0, 5, 0] (clamp negatives)
    ASSERT_EQ(result.outputs.size(), 1u);
    ASSERT_EQ(result.outputs[0].size(), 4u);
    EXPECT_FLOAT_EQ(result.outputs[0][0], 2.0f);
    EXPECT_FLOAT_EQ(result.outputs[0][1], 0.0f);  // ReLU(-2) = 0
    EXPECT_FLOAT_EQ(result.outputs[0][2], 5.0f);
    EXPECT_FLOAT_EQ(result.outputs[0][3], 0.0f);  // ReLU(-5) = 0
}

TEST(Executor, AddTwoInputs) {
    Graph g;
    auto a = g.add_input("a", {3});
    auto b = g.add_input("b", {3});
    auto add = g.add_node(std::make_shared<AddOp>(), {a, b});
    g.mark_output(add);

    Executor exec(g);
    std::vector<float> va = {1, 2, 3};
    std::vector<float> vb = {4, 5, 6};
    exec.set_input(0, va);
    exec.set_input(1, vb);
    auto result = exec.execute();

    EXPECT_EQ(result.outputs[0], (std::vector<float>{5, 7, 9}));
}

TEST(Executor, Profiling) {
    Graph g;
    auto in = g.add_input("x", {1024});
    auto relu = g.add_node(std::make_shared<ReLUOp>(), {in});
    g.mark_output(relu);

    Executor exec(g);
    std::vector<float> input(1024, 1.0f);
    exec.set_input(0, input);
    auto result = exec.execute(/*profile=*/true);

    EXPECT_GT(result.total_time_us, 0.0);
    EXPECT_EQ(result.nodes_executed, 1u);
    EXPECT_GT(result.peak_memory_bytes, 0u);
}

TEST(Executor, MultipleExecutions) {
    // Verify arena reset works — run twice, same result
    Graph g;
    auto in = g.add_input("x", {3});
    auto relu = g.add_node(std::make_shared<ReLUOp>(), {in});
    g.mark_output(relu);

    Executor exec(g);
    std::vector<float> input = {-1, 0, 1};
    exec.set_input(0, input);

    auto r1 = exec.execute();
    auto r2 = exec.execute();
    EXPECT_EQ(r1.outputs[0], r2.outputs[0]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Optimizer Tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST(Optimizer, DeadNodeElimination) {
    Graph g;
    auto in = g.add_input("x", {4});
    auto relu = g.add_node(std::make_shared<ReLUOp>(), {in});
    auto dead = g.add_node(std::make_shared<ReLUOp>(), {in}); // No consumers
    g.mark_output(relu);

    size_t removed = eliminate_dead_nodes(g);
    EXPECT_EQ(removed, 1u);
    EXPECT_TRUE(g.node(dead).is_dead());
    EXPECT_FALSE(g.node(relu).is_dead());
}

TEST(Optimizer, MatMulReLUFusion) {
    Graph g;
    auto in = g.add_input("input", {4, 8});
    auto w = g.add_input("weights", {8, 16});
    auto mm = g.add_node(std::make_shared<MatMulOp>(), {in, w});
    auto relu = g.add_node(std::make_shared<ReLUOp>(), {mm});
    g.mark_output(relu);

    EXPECT_EQ(g.live_node_count(), 4u);

    size_t fused = fuse_matmul_activation(g);
    EXPECT_EQ(fused, 1u);

    // MatMul should be dead, ReLU node should now be FusedMatMulReLU
    EXPECT_TRUE(g.node(mm).is_dead());
    EXPECT_EQ(g.node(relu).op_type(), OpType::FusedMatMulReLU);
    // Output shape unchanged
    EXPECT_EQ(g.node(relu).output_shape(), (std::vector<size_t>{4, 16}));
}

TEST(Optimizer, MatMulGELUFusion) {
    Graph g;
    auto in = g.add_input("input", {2, 4});
    auto w = g.add_input("weights", {4, 8});
    auto mm = g.add_node(std::make_shared<MatMulOp>(), {in, w});
    auto gelu = g.add_node(std::make_shared<GELUOp>(), {mm});
    g.mark_output(gelu);

    size_t fused = fuse_matmul_activation(g);
    EXPECT_EQ(fused, 1u);
    EXPECT_EQ(g.node(gelu).op_type(), OpType::FusedMatMulGELU);
}

TEST(Optimizer, NoFusionWhenMultipleConsumers) {
    // MatMul output used by BOTH ReLU and another Add → can't fuse
    Graph g;
    auto in = g.add_input("input", {4, 8});
    auto w = g.add_input("weights", {8, 16});
    auto mm = g.add_node(std::make_shared<MatMulOp>(), {in, w});
    auto relu = g.add_node(std::make_shared<ReLUOp>(), {mm});
    auto relu2 = g.add_node(std::make_shared<ReLUOp>(), {mm}); // Second consumer
    g.mark_output(relu);
    g.mark_output(relu2);

    size_t fused = fuse_matmul_activation(g);
    EXPECT_EQ(fused, 0u); // No fusion possible
}

TEST(Optimizer, FullOptimize) {
    Graph g;
    auto in = g.add_input("input", {4, 8});
    auto w = g.add_input("weights", {8, 16});
    auto mm = g.add_node(std::make_shared<MatMulOp>(), {in, w});
    auto relu = g.add_node(std::make_shared<ReLUOp>(), {mm});
    auto dead = g.add_node(std::make_shared<ReLUOp>(), {in}); // Dead
    g.mark_output(relu);

    auto stats = optimize(g);
    EXPECT_GE(stats.dead_nodes_removed, 1u);
    EXPECT_EQ(stats.fusions_applied, 1u);
    EXPECT_LT(stats.nodes_after, stats.nodes_before);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fusion Correctness: verify fused output matches unfused
// ═══════════════════════════════════════════════════════════════════════════════

TEST(FusionCorrectness, MatMulReLUMatchesUnfused) {
    // Build two graphs: one unfused, one fused. Same inputs → same output.
    std::vector<float> input = {1, -1, 2, -2, 3, -3}; // 2×3
    std::vector<float> weights = {1, 0.5, -1, 0.5, 1, -1}; // 3×2

    // Unfused graph
    Graph g1;
    auto in1 = g1.add_input("input", {2, 3});
    auto w1 = g1.add_input("weights", {3, 2});
    auto mm1 = g1.add_node(std::make_shared<MatMulOp>(), {in1, w1});
    auto relu1 = g1.add_node(std::make_shared<ReLUOp>(), {mm1});
    g1.mark_output(relu1);

    Executor exec1(g1);
    exec1.set_input(0, input);
    exec1.set_input(1, weights);
    auto r1 = exec1.execute();

    // Fused graph
    Graph g2;
    auto in2 = g2.add_input("input", {2, 3});
    auto w2 = g2.add_input("weights", {3, 2});
    auto mm2 = g2.add_node(std::make_shared<MatMulOp>(), {in2, w2});
    auto relu2 = g2.add_node(std::make_shared<ReLUOp>(), {mm2});
    g2.mark_output(relu2);
    optimize(g2); // Should fuse

    Executor exec2(g2);
    exec2.set_input(0, input);
    exec2.set_input(1, weights);
    auto r2 = exec2.execute();

    // Results must match
    ASSERT_EQ(r1.outputs[0].size(), r2.outputs[0].size());
    for (size_t i = 0; i < r1.outputs[0].size(); ++i) {
        EXPECT_NEAR(r1.outputs[0][i], r2.outputs[0][i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

TEST(FusionCorrectness, MemorySaved) {
    Graph g;
    auto in = g.add_input("input", {32, 768});
    auto w = g.add_input("weights", {768, 3072});
    auto mm = g.add_node(std::make_shared<MatMulOp>(), {in, w});
    auto relu = g.add_node(std::make_shared<ReLUOp>(), {mm});
    g.mark_output(relu);

    size_t mem_before = g.total_intermediate_bytes();
    auto stats = optimize(g);
    size_t mem_after = g.total_intermediate_bytes();

    // Fusion eliminates the MatMul intermediate (32×3072×4 = 384 KB)
    EXPECT_GT(stats.memory_saved_bytes, 0u);
    EXPECT_LT(mem_after, mem_before);
}
