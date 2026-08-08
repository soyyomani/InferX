#pragma once

#include <inferx/core/tracer.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <cmath>

namespace inferx::nn {

/// Simple BPE-like tokenizer for educational purposes.
/// Shows step-by-step how text becomes token IDs that a model can process.
class Tokenizer {
public:
    struct Token {
        std::string text;
        int id;
    };

    Tokenizer() {
        build_vocabulary();
    }

    /// Tokenize a string into token IDs with full tracing
    std::vector<int> encode(const std::string& text) {
        auto& tracer = core::Tracer::instance();

        // Step 1: Character-level split
        tracer.record("Tokenizer", "Step 1: Character Split",
            "Split input text into individual characters",
            {"Input: \"" + text + "\"",
             "Length: " + std::to_string(text.size()) + " characters",
             "Each character becomes a candidate for merging"});

        // Step 2: Whitespace pre-tokenization
        auto words = split_whitespace(text);
        std::vector<std::string> word_strs;
        for (auto& w : words) word_strs.push_back("\"" + w + "\"");

        tracer.record("Tokenizer", "Step 2: Whitespace Pre-tokenization",
            "Split text on whitespace boundaries into words",
            {"Words found: " + std::to_string(words.size()),
             "Result: [" + join(word_strs, ", ") + "]",
             "Each word is processed independently for BPE merges"});

        // Step 3: BPE merges (simplified)
        std::vector<Token> tokens;
        std::vector<std::string> merge_log;

        for (const auto& word : words) {
            auto word_tokens = bpe_encode_word(word, merge_log);
            tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
        }

        tracer.record("Tokenizer", "Step 3: BPE Merge Pairs",
            "Repeatedly merge the most frequent adjacent pair of symbols",
            merge_log.empty() ?
                std::vector<std::string>{"No merges needed - all tokens found in vocabulary"} :
                merge_log);

        // Step 4: Vocabulary lookup
        std::vector<int> ids;
        std::vector<std::string> lookup_log;
        for (const auto& tok : tokens) {
            ids.push_back(tok.id);
            lookup_log.push_back("\"" + tok.text + "\" → ID " + std::to_string(tok.id));
        }

        tracer.record("Tokenizer", "Step 4: Vocabulary ID Lookup",
            "Map each token string to its integer ID in the vocabulary table",
            lookup_log);

        // Step 5: Final output
        std::vector<std::string> id_strs;
        for (int id : ids) id_strs.push_back(std::to_string(id));

        tracer.record("Tokenizer", "Step 5: Final Token IDs",
            "The sequence of integers that the neural network receives",
            {"Token count: " + std::to_string(ids.size()),
             "IDs: [" + join(id_strs, ", ") + "]",
             "Each ID indexes into an embedding matrix to get a dense vector",
             "Vocabulary size: " + std::to_string(vocab_.size()) + " tokens"});

        return ids;
    }

    /// Decode token IDs back to text
    std::string decode(const std::vector<int>& ids) {
        std::string result;
        for (int id : ids) {
            if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
                result += id_to_token_[id];
            }
        }
        return result;
    }

    int vocab_size() const { return static_cast<int>(vocab_.size()); }

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;

    void build_vocabulary() {
        // Build a simple vocabulary: individual chars + common subwords
        int id = 0;

        // Special tokens
        add_token("<pad>", id++);
        add_token("<unk>", id++);
        add_token("<bos>", id++);
        add_token("<eos>", id++);

        // Single characters (ASCII printable)
        for (char c = ' '; c <= '~'; c++) {
            add_token(std::string(1, c), id++);
        }

        // Common English subwords (BPE-style merges result)
        std::vector<std::string> subwords = {
            "the", "ing", "tion", "er", "ed", "es", "al", "en",
            "re", "on", "an", "or", "is", "it", "at", "ar",
            "he", "ha", "th", "in", "to", "of", "and", "for",
            "are", "but", "not", "you", "all", "can", "had", "her",
            "was", "one", "our", "out", "with", "have", "this", "from",
            "they", "been", "said", "each", "which", "their", "will",
            "way", "about", "many", "then", "them", "would", "like",
            "Hello", "hello", "World", "world", "How", "how", "What",
            "what", "The", "This", "that", "when", "where", "who",
            "why", "AI", "model", "data", "input", "output", "layer",
            "attention", "transformer", "neural", "network", "token",
            "embedding", "softmax", "matrix", "vector", "weight",
        };

        for (const auto& sw : subwords) {
            if (vocab_.find(sw) == vocab_.end()) {
                add_token(sw, id++);
            }
        }
    }

    void add_token(const std::string& token, int id) {
        vocab_[token] = id;
        if (static_cast<int>(id_to_token_.size()) <= id) {
            id_to_token_.resize(id + 1);
        }
        id_to_token_[id] = token;
    }

    std::vector<std::string> split_whitespace(const std::string& text) {
        std::vector<std::string> words;
        std::string current;
        for (char c : text) {
            if (c == ' ' || c == '\t' || c == '\n') {
                if (!current.empty()) {
                    words.push_back(current);
                    current.clear();
                }
                words.push_back(std::string(1, c));
            } else {
                current += c;
            }
        }
        if (!current.empty()) words.push_back(current);
        return words;
    }

    std::vector<Token> bpe_encode_word(const std::string& word, std::vector<std::string>& merge_log) {
        // Check if whole word is in vocab
        if (vocab_.find(word) != vocab_.end()) {
            return {{word, vocab_[word]}};
        }

        // Try to find longest matching subwords (greedy left-to-right)
        std::vector<Token> tokens;
        size_t i = 0;
        while (i < word.size()) {
            size_t best_len = 1;
            int best_id = vocab_.count(std::string(1, word[i])) ?
                vocab_[std::string(1, word[i])] : 1; // <unk>

            // Find longest matching token starting at position i
            for (size_t len = std::min(word.size() - i, size_t(12)); len > 1; len--) {
                std::string sub = word.substr(i, len);
                if (vocab_.find(sub) != vocab_.end()) {
                    best_len = len;
                    best_id = vocab_[sub];
                    if (tokens.size() > 0 || i > 0) {
                        merge_log.push_back("Merge: found \"" + sub + "\" in vocabulary (ID " +
                            std::to_string(best_id) + ")");
                    }
                    break;
                }
            }

            std::string tok_text = word.substr(i, best_len);
            tokens.push_back({tok_text, best_id});
            i += best_len;
        }

        return tokens;
    }

    std::string join(const std::vector<std::string>& v, const std::string& sep) {
        std::string result;
        for (size_t i = 0; i < v.size(); i++) {
            if (i > 0) result += sep;
            result += v[i];
        }
        return result;
    }
};

} // namespace inferx::nn
