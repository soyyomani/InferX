#pragma once

/// @file operator.h
/// @brief Operator base class and concrete operator implementations for the graph.
///
/// In AI compilers/runtimes, an "operator" (or "kernel") is a single computation:
///   - MatMul: C = A × B
///   - ReLU: y = max(0, x)
///   - Add: z = x + y
///   - Softmax: y = exp(x) / sum(exp(x))
///
/// The key design pattern is separation of concerns:
///   - The Operator defines WHAT computation to perform
///   - The Graph defines the DATAFLOW between operators
///   - The Executor decides WHEN and WHERE to run each operator
///   - The Optimizer transforms the graph for better performance
///
/// This is the same architecture used by:
///   - TensorRT: ILayer → INetworkDefinition → IBuilder (optimize) → ICudaEngine
///   - ONNX Runtime: Op → Graph → GraphOptimizer → ExecutionProvider
///   - XLA: HloInstruction → HloComputation → HloModule → passes → codegen
///   - TVM: relay.Expr → relay.Function → passes → te.Schedule → build
///
/// Each operator:
///   1. Declares its input/output count and shapes (for validation)
///   2. Implements execute() which reads input buffers and writes output buffers
///   3. Has a name and type for debugging / fusion pattern matching

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace inferx::graph {

/// Tensor descriptor: shape + data pointer for graph execution.
/// This is a lightweight non-owning view — the Executor owns the memory.
struct TensorDesc {
    float* data = nullptr;
    std::vector<size_t> shape;

    [[nodiscard]] size_t numel() const noexcept {
        if (shape.empty()) return 0;
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    [[nodiscard]] size_t size_bytes() const noexcept {
        return numel() * sizeof(float);
    }
};

/// Operator type enumeration — used for pattern matching in fusion passes.
enum class OpType {
    // Elementwise
    ReLU,
    GELU,
    Sigmoid,
    Add,
    Mul,

    // Linear algebra
    MatMul,

    // Normalization
    Softmax,
    LayerNorm,

    // Fused (created by optimizer)
    FusedMatMulReLU,
    FusedMatMulGELU,

    // Special
    Identity,   // Pass-through (useful for graph manipulation)
    Constant,   // Produces a constant tensor
};

/// Convert OpType to string for debugging
inline std::string op_type_name(OpType type) {
    switch (type) {
        case OpType::ReLU:           return "ReLU";
        case OpType::GELU:           return "GELU";
        case OpType::Sigmoid:        return "Sigmoid";
        case OpType::Add:            return "Add";
        case OpType::Mul:            return "Mul";
        case OpType::MatMul:         return "MatMul";
        case OpType::Softmax:        return "Softmax";
        case OpType::LayerNorm:      return "LayerNorm";
        case OpType::FusedMatMulReLU: return "FusedMatMulReLU";
        case OpType::FusedMatMulGELU: return "FusedMatMulGELU";
        case OpType::Identity:       return "Identity";
        case OpType::Constant:       return "Constant";
    }
    return "Unknown";
}

/// Check if an operator is elementwise (operates independently per element)
inline bool is_elementwise(OpType type) noexcept {
    switch (type) {
        case OpType::ReLU:
        case OpType::GELU:
        case OpType::Sigmoid:
        case OpType::Add:
        case OpType::Mul:
            return true;
        default:
            return false;
    }
}

// ─── Operator Base Class ─────────────────────────────────────────────────────

/// Abstract base class for all operators.
///
/// Contract:
///   - num_inputs() / num_outputs(): declare I/O count
///   - infer_output_shape(): given input shapes, compute output shape
///   - execute(): perform the computation (reads inputs, writes outputs)
///
/// Operators are stateless — weights are passed as inputs, not stored internally.
/// This makes the graph serializable and operators reusable across different models.
class Operator {
public:
    virtual ~Operator() = default;

    /// Operator type (for pattern matching in optimization passes)
    [[nodiscard]] virtual OpType type() const noexcept = 0;

    /// Human-readable name
    [[nodiscard]] virtual std::string name() const { return op_type_name(type()); }

    /// Number of inputs this operator expects
    [[nodiscard]] virtual size_t num_inputs() const noexcept = 0;

    /// Number of outputs this operator produces
    [[nodiscard]] virtual size_t num_outputs() const noexcept { return 1; }

    /// Infer output shape from input shapes (static shape inference).
    /// This is called during graph construction for validation.
    [[nodiscard]] virtual std::vector<size_t> infer_output_shape(
        const std::vector<std::vector<size_t>>& input_shapes) const = 0;

    /// Execute the operator: read from inputs, write to outputs.
    /// All buffers are pre-allocated by the executor.
    virtual void execute(const std::vector<TensorDesc>& inputs,
                         std::vector<TensorDesc>& outputs) const = 0;
};

// ─── Concrete Operators ──────────────────────────────────────────────────────

/// ReLU: y[i] = max(0, x[i])
/// Elementwise, shape-preserving. The simplest non-linearity.
class ReLUOp : public Operator {
public:
    [[nodiscard]] OpType type() const noexcept override { return OpType::ReLU; }
    [[nodiscard]] size_t num_inputs() const noexcept override { return 1; }

    [[nodiscard]] std::vector<size_t> infer_output_shape(
        const std::vector<std::vector<size_t>>& input_shapes) const override {
        return input_shapes[0]; // Shape-preserving
    }

    void execute(const std::vector<TensorDesc>& inputs,
                 std::vector<TensorDesc>& outputs) const override {
        const auto& in = inputs[0];
        auto& out = outputs[0];
        size_t n = in.numel();
        for (size_t i = 0; i < n; ++i) {
            out.data[i] = std::max(0.0f, in.data[i]);
        }
    }
};

/// GELU: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
/// Used in GPT-2, BERT, most modern transformers. More expensive than ReLU.
class GELUOp : public Operator {
public:
    [[nodiscard]] OpType type() const noexcept override { return OpType::GELU; }
    [[nodiscard]] size_t num_inputs() const noexcept override { return 1; }

    [[nodiscard]] std::vector<size_t> infer_output_shape(
        const std::vector<std::vector<size_t>>& input_shapes) const override {
        return input_shapes[0];
    }

    void execute(const std::vector<TensorDesc>& inputs,
                 std::vector<TensorDesc>& outputs) const override {
        const auto& in = inputs[0];
        auto& out = outputs[0];
        constexpr float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
        constexpr float coeff = 0.044715f;

        size_t n = in.numel();
        for (size_t i = 0; i < n; ++i) {
            float x = in.data[i];
            float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            out.data[i] = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }
};

/// Sigmoid: y = 1 / (1 + exp(-x))
class SigmoidOp : public Operator {
public:
    [[nodiscard]] OpType type() const noexcept override { return OpType::Sigmoid; }
    [[nodiscard]] size_t num_inputs() const noexcept override { return 1; }

    [[nodiscard]] std::vector<size_t> infer_output_shape(
        const std::vector<std::vector<size_t>>& input_shapes) const override {
        return input_shapes[0];
    }

    void execute(const std::vector<TensorDesc>& inputs,
                 std::vector<TensorDesc>& outputs) const override {
        const auto& in = inputs[0];
        auto& out = outputs[0];
        size_t n = in.numel();
        for (size_t i = 0; i < n; ++i) {
            out.data[i] = 1.0f / (1.0f + std::exp(-in.data[i]));
        }
    }
};

/// Add: z = x + y (elementwise, broadcasting not implemented for simplicity)
class AddOp : public Operator {
public:
    [[nodiscard]] OpType type() const noexcept override { return OpType::Add; }
    [[nodiscard]] size_t num_inputs() const noexcept override { return 2; }

    [[nodiscard]] std::vector<size_t> infer_output_shape(
        const std::vector<std::vector<size_t>>& input_shapes) const override {
        // Both inputs must have same shape
        if (input_shapes[0] != input_shapes[1]) {
            throw std::invalid_argument("Add: input shapes must match");
        }
        return input_shapes[0];
    }

    void execute(const std::vector<TensorDesc>& inputs,
                 std::vector<TensorDesc>& outputs) const override {
        const auto& a = inputs[0];
        const auto& b = inputs[1];
        auto& out = outputs[0];
        size_t n = a.numel();
        for (size_t i = 0; i < n; ++i) {
            out.data[i] = a.data[i] + b.data[i];
        }
    }
};

/// Mul: z = x * y (elementwise)
class MulOp : public Operator {
public:
    [[nodiscard]] OpType type() const noexcept override { return OpType::Mul; }
    [[nodiscard]] size_t num_inputs() const noexcept override { return 2; }

    [[nodiscard]] std::vector<size_t> infer_output_shape(
        const std::vector<std::vector<size_t>>& input_shapes) const override {
        if (input_shapes[0] != input_shapes[1]) {
            throw std::invalid_argument("Mul: input shapes must match");
        }
        return input_shapes[0];
    }

    void execute(const std::vector<TensorDesc>& inputs,
                 std::vector<TensorDesc>& outputs) const override {
        const auto& a = inputs[0];
        const auto& b = inputs[1];
        auto& out = outputs[0];
        size_t n = a.numel();
        for (size_t i = 0; i < n; ++i) {
            out.data[i] = a.data[i] * b.data[i];
        }
    }
};

/// MatMul: C = A × B
/// Input 0: [M × K], Input 1: [K × N], Output: [M × N]
/// Uses the optimized kernel from inferx::kernels when available.
class MatMulOp : public Operator {
public:
    [[nodiscard]] OpType type() const noexcept override { return OpType::MatMul; }
    [[nodiscard]] size_t num_inputs() const noexcept override { return 2; }

    [[nodiscard]] std::vector<size_t> infer_output_shape(
        const std::vector<std::vector<size_t>>& input_shapes) const override {
        const auto& a_shape = input_shapes[0]; // [M, K]
        const auto& b_shape = input_shapes[1]; // [K, N]

        if (a_shape.size() != 2 || b_shape.size() != 2) {
            throw std::invalid_argument("MatMul: inputs must be 2D matrices");
        }
        if (a_shape[1] != b_shape[0]) {
            throw std::invalid_argument(
                "MatMul: inner dimensions must match (A: " +
                std::to_string(a_shape[1]) + " vs B: " + std::to_string(b_shape[0]) + ")");
        }
        return {a_shape[0], b_shape[1]}; // [M, N]
    }

    void execute(const std::vector<TensorDesc>& inputs,
                 std::vector<TensorDesc>& outputs) const override {
        const auto& A = inputs[0];
        const auto& B = inputs[1];
        auto& C = outputs[0];

        size_t M = A.shape[0];
        size_t K = A.shape[1];
        size_t N = B.shape[1];

        // Zero output
        std::memset(C.data, 0, M * N * sizeof(float));

        // ikj loop order for better cache behavior
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float a_ik = A.data[i * K + k];
                for (size_t j = 0; j < N; ++j) {
                    C.data[i * N + j] += a_ik * B.data[k * N + j];
                }
            }
        }
    }
};

/// Softmax: y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
/// Numerically stable implementation. Operates on last dimension.
class SoftmaxOp : public Operator {
public:
    [[nodiscard]] OpType type() const noexcept override { return OpType::Softmax; }
    [[nodiscard]] size_t num_inputs() const noexcept override { return 1; }

    [[nodiscard]] std::vector<size_t> infer_output_shape(
        const std::vector<std::vector<size_t>>& input_shapes) const override {
        return input_shapes[0]; // Shape-preserving
    }

    void execute(const std::vector<TensorDesc>& inputs,
                 std::vector<TensorDesc>& outputs) const override {
        const auto& in = inputs[0];
        auto& out = outputs[0];

        if (in.shape.size() == 1) {
            // 1D: softmax over entire vector
            softmax_1d(in.data, out.data, in.shape[0]);
        } else if (in.shape.size() == 2) {
            // 2D: softmax over each row
            size_t rows = in.shape[0];
            size_t cols = in.shape[1];
            for (size_t r = 0; r < rows; ++r) {
                softmax_1d(in.data + r * cols, out.data + r * cols, cols);
            }
        }
    }

private:
    static void softmax_1d(const float* in, float* out, size_t n) {
        float max_val = *std::max_element(in, in + n);
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            out[i] = std::exp(in[i] - max_val);
            sum += out[i];
        }
        for (size_t i = 0; i < n; ++i) {
            out[i] /= sum;
        }
    }
};

/// Identity: y = x (pass-through, useful for graph rewrites)
class IdentityOp : public Operator {
public:
    [[nodiscard]] OpType type() const noexcept override { return OpType::Identity; }
    [[nodiscard]] size_t num_inputs() const noexcept override { return 1; }

    [[nodiscard]] std::vector<size_t> infer_output_shape(
        const std::vector<std::vector<size_t>>& input_shapes) const override {
        return input_shapes[0];
    }

    void execute(const std::vector<TensorDesc>& inputs,
                 std::vector<TensorDesc>& outputs) const override {
        const auto& in = inputs[0];
        auto& out = outputs[0];
        std::memcpy(out.data, in.data, in.size_bytes());
    }
};

// ─── Fused Operators (created by optimizer) ──────────────────────────────────

/// FusedMatMulReLU: C = max(0, A × B)
/// Eliminates the intermediate buffer between MatMul and ReLU.
/// Memory savings: one fewer tensor allocation (M×N floats).
/// Latency savings: one fewer memory pass over M×N elements.
///
/// This is the most common fusion in inference runtimes:
///   - TensorRT: convolution + bias + activation fusion
///   - oneDNN: matmul + eltwise post-op fusion
///   - XLA: dot + maximum fusion
class FusedMatMulReLUOp : public Operator {
public:
    [[nodiscard]] OpType type() const noexcept override { return OpType::FusedMatMulReLU; }
    [[nodiscard]] size_t num_inputs() const noexcept override { return 2; }

    [[nodiscard]] std::vector<size_t> infer_output_shape(
        const std::vector<std::vector<size_t>>& input_shapes) const override {
        const auto& a_shape = input_shapes[0];
        const auto& b_shape = input_shapes[1];
        if (a_shape.size() != 2 || b_shape.size() != 2) {
            throw std::invalid_argument("FusedMatMulReLU: inputs must be 2D");
        }
        if (a_shape[1] != b_shape[0]) {
            throw std::invalid_argument("FusedMatMulReLU: inner dims must match");
        }
        return {a_shape[0], b_shape[1]};
    }

    void execute(const std::vector<TensorDesc>& inputs,
                 std::vector<TensorDesc>& outputs) const override {
        const auto& A = inputs[0];
        const auto& B = inputs[1];
        auto& C = outputs[0];

        size_t M = A.shape[0];
        size_t K = A.shape[1];
        size_t N = B.shape[1];

        // Fused: compute matmul and apply ReLU in one pass
        // No intermediate buffer needed!
        std::memset(C.data, 0, M * N * sizeof(float));

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float a_ik = A.data[i * K + k];
                for (size_t j = 0; j < N; ++j) {
                    C.data[i * N + j] += a_ik * B.data[k * N + j];
                }
            }
            // Apply ReLU to the completed row
            for (size_t j = 0; j < N; ++j) {
                C.data[i * N + j] = std::max(0.0f, C.data[i * N + j]);
            }
        }
    }
};

/// FusedMatMulGELU: C = GELU(A × B)
/// Same principle as FusedMatMulReLU but with GELU activation.
class FusedMatMulGELUOp : public Operator {
public:
    [[nodiscard]] OpType type() const noexcept override { return OpType::FusedMatMulGELU; }
    [[nodiscard]] size_t num_inputs() const noexcept override { return 2; }

    [[nodiscard]] std::vector<size_t> infer_output_shape(
        const std::vector<std::vector<size_t>>& input_shapes) const override {
        const auto& a_shape = input_shapes[0];
        const auto& b_shape = input_shapes[1];
        if (a_shape.size() != 2 || b_shape.size() != 2) {
            throw std::invalid_argument("FusedMatMulGELU: inputs must be 2D");
        }
        if (a_shape[1] != b_shape[0]) {
            throw std::invalid_argument("FusedMatMulGELU: inner dims must match");
        }
        return {a_shape[0], b_shape[1]};
    }

    void execute(const std::vector<TensorDesc>& inputs,
                 std::vector<TensorDesc>& outputs) const override {
        const auto& A = inputs[0];
        const auto& B = inputs[1];
        auto& C = outputs[0];

        size_t M = A.shape[0];
        size_t K = A.shape[1];
        size_t N = B.shape[1];

        constexpr float sqrt_2_over_pi = 0.7978845608f;
        constexpr float coeff = 0.044715f;

        std::memset(C.data, 0, M * N * sizeof(float));

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float a_ik = A.data[i * K + k];
                for (size_t j = 0; j < N; ++j) {
                    C.data[i * N + j] += a_ik * B.data[k * N + j];
                }
            }
            // Apply GELU to the completed row
            for (size_t j = 0; j < N; ++j) {
                float x = C.data[i * N + j];
                float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
                C.data[i * N + j] = 0.5f * x * (1.0f + std::tanh(inner));
            }
        }
    }
};

// ─── Operator Factory ────────────────────────────────────────────────────────

/// Create an operator by type. Used when building graphs from serialized format.
inline std::shared_ptr<Operator> make_operator(OpType type) {
    switch (type) {
        case OpType::ReLU:           return std::make_shared<ReLUOp>();
        case OpType::GELU:           return std::make_shared<GELUOp>();
        case OpType::Sigmoid:        return std::make_shared<SigmoidOp>();
        case OpType::Add:            return std::make_shared<AddOp>();
        case OpType::Mul:            return std::make_shared<MulOp>();
        case OpType::MatMul:         return std::make_shared<MatMulOp>();
        case OpType::Softmax:        return std::make_shared<SoftmaxOp>();
        case OpType::Identity:       return std::make_shared<IdentityOp>();
        case OpType::FusedMatMulReLU: return std::make_shared<FusedMatMulReLUOp>();
        case OpType::FusedMatMulGELU: return std::make_shared<FusedMatMulGELUOp>();
        default:
            throw std::invalid_argument("Unknown operator type: " + op_type_name(type));
    }
}

} // namespace inferx::graph
