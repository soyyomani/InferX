#pragma once

#include <inferx/core/tracer.h>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace inferx::nn {

/// Self-Attention mechanism with step-by-step tracing.
/// Shows Q/K/V projection, scaled dot-product attention, and output.
class Attention {
public:
    Attention(int embed_dim, int num_heads = 1)
        : embed_dim_(embed_dim), num_heads_(num_heads),
          head_dim_(embed_dim / num_heads) {
        // Initialize Q, K, V weight matrices with deterministic values
        int total = embed_dim * embed_dim;
        wq_.resize(total);
        wk_.resize(total);
        wv_.resize(total);
        wo_.resize(total);

        std::mt19937 gen(123);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (int i = 0; i < total; i++) {
            wq_[i] = dist(gen);
            wk_[i] = dist(gen);
            wv_[i] = dist(gen);
            wo_[i] = dist(gen);
        }
    }

    /// Forward pass: self-attention on input [seq_len × embed_dim]
    std::vector<float> forward(const std::vector<float>& input, int seq_len) {
        auto& tracer = core::Tracer::instance();

        // Step 1: Explain attention concept
        tracer.record("Attention", "Step 1: Self-Attention Overview",
            "Let each token look at all other tokens to understand context",
            {"Input shape: [" + std::to_string(seq_len) + " × " +
                std::to_string(embed_dim_) + "]",
             "Number of heads: " + std::to_string(num_heads_),
             "Head dimension: " + std::to_string(head_dim_),
             "",
             "Intuition: 'Bank' means different things in:",
             "  'I went to the river bank' (nature)",
             "  'I went to the bank to deposit' (finance)",
             "Attention lets each word look at context to disambiguate."});

        // Step 2: Q, K, V projections
        auto Q = matmul(input, wq_, seq_len, embed_dim_, embed_dim_);
        auto K = matmul(input, wk_, seq_len, embed_dim_, embed_dim_);
        auto V = matmul(input, wv_, seq_len, embed_dim_, embed_dim_);

        std::vector<std::string> qkv_details;
        qkv_details.push_back("Q = Input × W_Q  (What am I looking for?)");
        qkv_details.push_back("K = Input × W_K  (What do I contain?)");
        qkv_details.push_back("V = Input × W_V  (What info do I give?)");
        qkv_details.push_back("");
        qkv_details.push_back("Shape: [" + std::to_string(seq_len) + "×" +
            std::to_string(embed_dim_) + "] × [" + std::to_string(embed_dim_) +
            "×" + std::to_string(embed_dim_) + "] = [" + std::to_string(seq_len) +
            "×" + std::to_string(embed_dim_) + "]");
        qkv_details.push_back("Total params: 3 × " +
            std::to_string(embed_dim_ * embed_dim_) + " = " +
            std::to_string(3 * embed_dim_ * embed_dim_));

        // Show sample Q values
        if (seq_len > 0) {
            std::ostringstream q_oss;
            q_oss << "Q[0] sample: [";
            for (int j = 0; j < std::min(embed_dim_, 4); j++) {
                if (j > 0) q_oss << ", ";
                q_oss << std::fixed << std::setprecision(4) << Q[j];
            }
            q_oss << ", ...]";
            qkv_details.push_back(q_oss.str());
        }

        tracer.record("Attention", "Step 2: Q, K, V Linear Projections",
            "Project input into Query, Key, Value spaces via learned matrices",
            qkv_details);

        // Step 3: Scaled dot-product attention scores
        float scale = std::sqrt(static_cast<float>(head_dim_));
        std::vector<float> scores(seq_len * seq_len, 0.0f);

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float dot = 0;
                for (int k = 0; k < head_dim_; k++) {
                    dot += Q[i * embed_dim_ + k] * K[j * embed_dim_ + k];
                }
                scores[i * seq_len + j] = dot / scale;
            }
        }

        std::vector<std::string> score_details;
        score_details.push_back("score[i][j] = (Q[i] · K[j]) / √d_k");
        score_details.push_back("√d_k = √" + std::to_string(head_dim_) +
            " = " + format_float(scale));
        score_details.push_back("Scaling prevents dot products from growing too large");
        score_details.push_back("");
        score_details.push_back("Attention score matrix [" +
            std::to_string(seq_len) + "×" + std::to_string(seq_len) + "]:");

        // Show score matrix (first few rows/cols)
        int show = std::min(seq_len, 4);
        for (int i = 0; i < show; i++) {
            std::ostringstream row_oss;
            row_oss << "  [";
            for (int j = 0; j < show; j++) {
                if (j > 0) row_oss << ", ";
                row_oss << std::fixed << std::setprecision(3)
                        << scores[i * seq_len + j];
            }
            if (seq_len > show) row_oss << ", ...";
            row_oss << "]";
            score_details.push_back(row_oss.str());
        }

        tracer.record("Attention", "Step 3: Scaled Dot-Product Scores",
            "Compute attention scores: how much each token attends to every other",
            score_details);

        // Step 4: Softmax over scores (row-wise)
        std::vector<float> attn_weights(seq_len * seq_len);
        for (int i = 0; i < seq_len; i++) {
            float max_val = *std::max_element(
                scores.begin() + i * seq_len,
                scores.begin() + (i + 1) * seq_len);
            float sum = 0;
            for (int j = 0; j < seq_len; j++) {
                attn_weights[i * seq_len + j] = std::exp(scores[i * seq_len + j] - max_val);
                sum += attn_weights[i * seq_len + j];
            }
            for (int j = 0; j < seq_len; j++) {
                attn_weights[i * seq_len + j] /= sum;
            }
        }

        std::vector<std::string> softmax_details;
        softmax_details.push_back("Apply softmax row-wise: each row becomes a probability distribution");
        softmax_details.push_back("attn_weights[i][j] = softmax(scores[i]) → P(token i attends to j)");
        softmax_details.push_back("");
        softmax_details.push_back("Attention weight matrix (probabilities):");
        for (int i = 0; i < show; i++) {
            std::ostringstream row_oss;
            row_oss << "  token " << i << " attends to: [";
            for (int j = 0; j < show; j++) {
                if (j > 0) row_oss << ", ";
                row_oss << std::fixed << std::setprecision(3)
                        << attn_weights[i * seq_len + j];
            }
            if (seq_len > show) row_oss << ", ...";
            row_oss << "]";
            softmax_details.push_back(row_oss.str());
        }
        softmax_details.push_back("");
        softmax_details.push_back("Each row sums to 1.0 (valid probability distribution)");

        tracer.record("Attention",
            "Step 4: Softmax → Attention Weights",
            "Convert scores to probabilities (how much to attend to each token)",
            softmax_details);

        // Step 5: Weighted sum of values
        std::vector<float> output(seq_len * embed_dim_, 0.0f);
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float w = attn_weights[i * seq_len + j];
                for (int k = 0; k < embed_dim_; k++) {
                    output[i * embed_dim_ + k] += w * V[j * embed_dim_ + k];
                }
            }
        }

        std::vector<std::string> output_details;
        output_details.push_back("output[i] = Σ_j attn_weights[i][j] × V[j]");
        output_details.push_back("Each token's output = weighted average of ALL value vectors");
        output_details.push_back("");
        output_details.push_back("Output shape: [" + std::to_string(seq_len) + " × " +
            std::to_string(embed_dim_) + "]");
        if (seq_len > 0) {
            std::ostringstream out_oss;
            out_oss << "output[0] = [";
            for (int j = 0; j < std::min(embed_dim_, 4); j++) {
                if (j > 0) out_oss << ", ";
                out_oss << std::fixed << std::setprecision(4) << output[j];
            }
            out_oss << ", ...]";
            output_details.push_back(out_oss.str());
        }
        output_details.push_back("");
        output_details.push_back("Each token now contains information from ALL other tokens,");
        output_details.push_back("weighted by how relevant they are (attention weights).");

        tracer.record("Attention",
            "Step 5: Weighted Value Aggregation",
            "Multiply attention weights by values to get context-aware representations",
            output_details);

        return output;
    }

    /// Get the attention weights from the last forward pass (for visualization)
    const std::vector<float>& last_attention_weights() const {
        return last_attn_weights_;
    }

private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    std::vector<float> wq_, wk_, wv_, wo_;
    std::vector<float> last_attn_weights_;

    std::vector<float> matmul(const std::vector<float>& A,
                              const std::vector<float>& B,
                              int M, int K, int N) {
        std::vector<float> C(M * N, 0.0f);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
        return C;
    }

    static std::string format_float(float v) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << v;
        return oss.str();
    }
};

} // namespace inferx::nn
