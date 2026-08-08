#pragma once

#include <inferx/core/tracer.h>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cassert>

namespace inferx::nn {

/// Matrix multiplication with step-by-step visualization of dot products.
/// Shows exactly how C = A × B works at the element level.
class MatMul {
public:
    /// Multiply matrices A[M×K] × B[K×N] = C[M×N] with full tracing
    static std::vector<float> forward(
        const std::vector<float>& A, int M, int K,
        const std::vector<float>& B, int K2, int N) {

        auto& tracer = core::Tracer::instance();
        assert(K == K2);

        // Step 1: Dimensions
        tracer.record("MatMul", "Step 1: Matrix Dimensions",
            "Verify shapes are compatible for multiplication",
            {"Matrix A shape: [" + std::to_string(M) + " × " + std::to_string(K) + "]",
             "Matrix B shape: [" + std::to_string(K) + " × " + std::to_string(N) + "]",
             "Rule: A columns (" + std::to_string(K) + ") must equal B rows (" + std::to_string(K2) + ") ✓",
             "Output C shape: [" + std::to_string(M) + " × " + std::to_string(N) + "]",
             "Total operations: " + std::to_string(M * N) + " dot products",
             "Each dot product: " + std::to_string(K) + " multiplications + " +
                std::to_string(K - 1) + " additions",
             "Total FLOPs: " + std::to_string(2L * M * N * K) + " (multiply-accumulate)"});

        // Step 2: Show the algorithm
        tracer.record("MatMul", "Step 2: The Algorithm",
            "C[i][j] = Σ(k=0 to K-1) A[i][k] × B[k][j]",
            {"For each element C[i][j] in the output:",
             "  1. Take row i from matrix A",
             "  2. Take column j from matrix B",
             "  3. Multiply corresponding elements pairwise",
             "  4. Sum all products → that's the dot product",
             "",
             "This is the CORE operation of neural networks.",
             "Every 'Linear layer' is just a matrix multiply + bias."});

        // Step 3: Compute with detailed traces for small matrices
        std::vector<float> C(M * N, 0.0f);
        int trace_limit = std::min(M * N, 4); // Only trace first few elements
        std::vector<std::string> compute_details;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                std::ostringstream dot_oss;

                if (i * N + j < trace_limit) {
                    dot_oss << "C[" << i << "][" << j << "] = ";
                }

                for (int k = 0; k < K; k++) {
                    float a_val = A[i * K + k];
                    float b_val = B[k * N + j];
                    float product = a_val * b_val;
                    sum += product;

                    if (i * N + j < trace_limit && k < 4) {
                        if (k > 0) dot_oss << " + ";
                        dot_oss << std::fixed << std::setprecision(3)
                                << a_val << "×" << b_val;
                    }
                }

                C[i * N + j] = sum;

                if (i * N + j < trace_limit) {
                    if (K > 4) dot_oss << " + ...";
                    dot_oss << " = " << std::fixed << std::setprecision(4) << sum;
                    compute_details.push_back(dot_oss.str());
                }
            }
        }

        if (M * N > trace_limit) {
            compute_details.push_back("... (" + std::to_string(M * N - trace_limit) +
                " more dot products computed)");
        }

        tracer.record("MatMul", "Step 3: Dot Product Computation",
            "Computing each element of the output matrix",
            compute_details);

        // Step 4: Result summary
        float max_val = *std::max_element(C.begin(), C.end());
        float min_val = *std::min_element(C.begin(), C.end());
        float sum_val = 0;
        for (float v : C) sum_val += v;

        tracer.record("MatMul", "Step 4: Result Matrix",
            "Output matrix C with shape [" + std::to_string(M) + " × " + std::to_string(N) + "]",
            {"Output elements: " + std::to_string(M * N),
             "Value range: [" + std::to_string(min_val) + ", " + std::to_string(max_val) + "]",
             "Sum: " + std::to_string(sum_val),
             "Memory: " + std::to_string(M * N * 4) + " bytes"});

        return C;
    }
};

} // namespace inferx::nn
