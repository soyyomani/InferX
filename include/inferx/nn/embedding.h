#pragma once

#include <inferx/core/tracer.h>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>

namespace inferx::nn {

/// Embedding layer - converts token IDs to dense vectors.
/// Shows the math of how a simple lookup table works + positional encoding.
class Embedding {
public:
    Embedding(int vocab_size, int embed_dim)
        : vocab_size_(vocab_size), embed_dim_(embed_dim) {
        // Initialize with deterministic pseudo-random weights
        weights_.resize(vocab_size * embed_dim);
        std::mt19937 gen(42); // fixed seed for reproducibility
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (auto& w : weights_) {
            w = dist(gen);
        }
    }

    /// Look up embeddings for a sequence of token IDs with full tracing
    std::vector<float> forward(const std::vector<int>& token_ids) {
        auto& tracer = core::Tracer::instance();
        int seq_len = static_cast<int>(token_ids.size());

        // Step 1: Explain what embedding does
        tracer.record("Embedding", "Step 1: Embedding Lookup Table",
            "Each token ID selects one row from the embedding matrix",
            {"Embedding matrix shape: [" + std::to_string(vocab_size_) + " × " +
                std::to_string(embed_dim_) + "]",
             "Each row is a learned " + std::to_string(embed_dim_) + "-dimensional vector",
             "Token ID acts as row index into this matrix",
             "This is the FIRST learnable layer - these vectors capture word meaning"});

        // Step 2: Show the actual lookups
        std::vector<float> output(seq_len * embed_dim_);
        std::vector<std::string> lookup_details;

        for (int i = 0; i < seq_len && i < 4; i++) { // Show first 4 for brevity
            int id = token_ids[i];
            int safe_id = (id >= 0 && id < vocab_size_) ? id : 0;

            // Copy embedding vector
            for (int j = 0; j < embed_dim_; j++) {
                output[i * embed_dim_ + j] = weights_[safe_id * embed_dim_ + j];
            }

            // Format first few values for display
            std::ostringstream oss;
            oss << "token_id=" << id << " → row[" << safe_id << "] = [";
            for (int j = 0; j < std::min(embed_dim_, 4); j++) {
                if (j > 0) oss << ", ";
                oss << std::fixed << std::setprecision(4)
                    << weights_[safe_id * embed_dim_ + j];
            }
            oss << ", ...]";
            lookup_details.push_back(oss.str());
        }
        if (seq_len > 4) {
            lookup_details.push_back("... (" + std::to_string(seq_len - 4) + " more lookups)");
        }

        // Fill remaining tokens
        for (int i = std::min(seq_len, 4); i < seq_len; i++) {
            int id = token_ids[i];
            int safe_id = (id >= 0 && id < vocab_size_) ? id : 0;
            for (int j = 0; j < embed_dim_; j++) {
                output[i * embed_dim_ + j] = weights_[safe_id * embed_dim_ + j];
            }
        }

        tracer.record("Embedding", "Step 2: Token → Vector Lookup",
            "For each token ID, fetch its corresponding row from the weight matrix",
            lookup_details);

        // Step 3: Positional encoding
        std::vector<std::string> pos_details;
        pos_details.push_back("Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d))");
        pos_details.push_back("Formula: PE(pos, 2i+1) = cos(pos / 10000^(2i/d))");
        pos_details.push_back("This encodes position information into the vector");

        for (int pos = 0; pos < seq_len; pos++) {
            for (int i = 0; i < embed_dim_; i++) {
                float angle = static_cast<float>(pos) /
                    std::pow(10000.0f, static_cast<float>(2 * (i / 2)) / embed_dim_);
                if (i % 2 == 0) {
                    output[pos * embed_dim_ + i] += std::sin(angle);
                } else {
                    output[pos * embed_dim_ + i] += std::cos(angle);
                }
            }

            if (pos < 3) {
                std::ostringstream oss;
                oss << "pos=" << pos << ": [";
                for (int j = 0; j < std::min(embed_dim_, 4); j++) {
                    if (j > 0) oss << ", ";
                    oss << std::fixed << std::setprecision(4)
                        << output[pos * embed_dim_ + j];
                }
                oss << ", ...] (after adding PE)";
                pos_details.push_back(oss.str());
            }
        }

        tracer.record("Embedding", "Step 3: Add Positional Encoding",
            "Add position-dependent signals so the model knows word ORDER",
            pos_details);

        // Step 4: Output shape
        tracer.record("Embedding", "Step 4: Output Embedding Matrix",
            "Final embedded representation ready for attention",
            {"Output shape: [" + std::to_string(seq_len) + " × " +
                std::to_string(embed_dim_) + "]",
             "Each row is now: token_meaning + position_info",
             "Total parameters in this layer: " +
                std::to_string(vocab_size_ * embed_dim_),
             "Memory: " + std::to_string(vocab_size_ * embed_dim_ * 4) + " bytes (float32)"});

        return output;
    }

    int embed_dim() const { return embed_dim_; }
    int vocab_size() const { return vocab_size_; }

private:
    int vocab_size_;
    int embed_dim_;
    std::vector<float> weights_;
};

} // namespace inferx::nn
