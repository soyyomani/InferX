#pragma once

#include <inferx/core/tracer.h>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace inferx::nn {

/// Softmax with numerical stability (max subtraction trick).
/// Traces every step: find max, subtract, exp, sum, normalize.
class Softmax {
public:
    /// Apply softmax to a vector of logits with full tracing
    static std::vector<float> forward(const std::vector<float>& logits) {
        auto& tracer = core::Tracer::instance();
        int n = static_cast<int>(logits.size());

        // Step 1: Raw logits
        std::vector<std::string> raw_details;
        raw_details.push_back("Input logits (" + std::to_string(n) + " values):");
        std::ostringstream oss;
        oss << "[";
        for (int i = 0; i < std::min(n, 8); i++) {
            if (i > 0) oss << ", ";
            oss << std::fixed << std::setprecision(3) << logits[i];
        }
        if (n > 8) oss << ", ...";
        oss << "]";
        raw_details.push_back(oss.str());
        raw_details.push_back("Problem: e^(large number) → overflow!");
        raw_details.push_back("Solution: subtract max first (doesn't change ratios)");

        tracer.record("Softmax", "Step 1: Input Logits (Raw Scores)",
            "Neural network outputs unnormalized scores called logits",
            raw_details);

        // Step 2: Find max (numerical stability)
        float max_val = *std::max_element(logits.begin(), logits.end());

        tracer.record("Softmax", "Step 2: Find Maximum (Stability Trick)",
            "max(logits) = " + format_float(max_val),
            {"Scan all " + std::to_string(n) + " values to find the maximum",
             "max = " + format_float(max_val),
             "",
             "WHY? Without this trick:",
             "  e^1000 = Infinity (overflow!)",
             "  But e^(1000-1000) = e^0 = 1 (safe!)",
             "",
             "Mathematical proof it's equivalent:",
             "  softmax(x)_i = e^(x_i) / Σe^(x_j)",
             "  = e^(x_i - max) / Σe^(x_j - max)",
             "  Dividing numerator and denominator by e^max cancels out"});

        // Step 3: Subtract max and exponentiate
        std::vector<float> shifted(n);
        std::vector<float> exps(n);
        std::vector<std::string> exp_details;

        exp_details.push_back("For each logit: compute e^(logit - max)");
        for (int i = 0; i < n; i++) {
            shifted[i] = logits[i] - max_val;
            exps[i] = std::exp(shifted[i]);

            if (i < 6) {
                std::ostringstream step_oss;
                step_oss << "e^(" << std::fixed << std::setprecision(3)
                         << logits[i] << " - " << max_val << ") = e^("
                         << shifted[i] << ") = " << std::setprecision(4) << exps[i];
                exp_details.push_back(step_oss.str());
            }
        }
        if (n > 6) exp_details.push_back("... (" + std::to_string(n - 6) + " more)");

        tracer.record("Softmax", "Step 3: Subtract Max & Exponentiate",
            "Shift all values down, then apply e^x (makes all values positive)",
            exp_details);

        // Step 4: Sum of exponentials
        float sum_exp = std::accumulate(exps.begin(), exps.end(), 0.0f);

        std::vector<std::string> sum_details;
        std::ostringstream sum_oss;
        sum_oss << "Σ = ";
        for (int i = 0; i < std::min(n, 4); i++) {
            if (i > 0) sum_oss << " + ";
            sum_oss << std::fixed << std::setprecision(4) << exps[i];
        }
        if (n > 4) sum_oss << " + ...";
        sum_oss << " = " << std::setprecision(4) << sum_exp;
        sum_details.push_back(sum_oss.str());
        sum_details.push_back("This sum becomes the denominator (normalizer)");

        tracer.record("Softmax", "Step 4: Sum of Exponentials",
            "Sum all e^(shifted) values to get the normalizing constant",
            sum_details);

        // Step 5: Normalize (divide by sum)
        std::vector<float> probs(n);
        std::vector<std::string> norm_details;
        norm_details.push_back("probability_i = e^(x_i - max) / Σe^(x_j - max)");
        norm_details.push_back("");

        for (int i = 0; i < n; i++) {
            probs[i] = exps[i] / sum_exp;

            if (i < 6) {
                std::ostringstream norm_oss;
                norm_oss << "P[" << i << "] = " << std::fixed << std::setprecision(4)
                         << exps[i] << " / " << sum_exp
                         << " = " << std::setprecision(4) << probs[i]
                         << " (" << std::setprecision(1) << (probs[i] * 100) << "%)";
                norm_details.push_back(norm_oss.str());
            }
        }
        if (n > 6) norm_details.push_back("...");

        // Verify sum = 1
        float prob_sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        norm_details.push_back("");
        norm_details.push_back("Verification: Σ probabilities = " + format_float(prob_sum) + " ≈ 1.0 ✓");

        tracer.record("Softmax", "Step 5: Normalize → Probabilities",
            "Divide each exp value by the sum to get valid probabilities (0-1, sum to 1)",
            norm_details);

        // Step 6: Properties of the output
        auto max_prob_it = std::max_element(probs.begin(), probs.end());
        int argmax = static_cast<int>(max_prob_it - probs.begin());
        float entropy = 0;
        for (float p : probs) {
            if (p > 1e-10f) entropy -= p * std::log2(p);
        }

        tracer.record("Softmax", "Step 6: Output Properties",
            "Probability distribution analysis",
            {"All values in [0, 1] ✓",
             "Sum = " + format_float(prob_sum) + " ≈ 1.0 ✓",
             "Argmax (most likely): index " + std::to_string(argmax) +
                " with P=" + format_float(probs[argmax]),
             "Entropy: " + format_float(entropy) + " bits",
             "  (Low entropy = confident, High entropy = uncertain)",
             "",
             "KEY INSIGHT: Softmax amplifies differences!",
             "  Small logit differences → large probability ratios",
             "  This is how the model becomes 'confident' about choices"});

        return probs;
    }

    /// Apply softmax to a 2D matrix (row-wise) with tracing
    static std::vector<float> forward_2d(const std::vector<float>& logits, int rows, int cols) {
        auto& tracer = core::Tracer::instance();

        tracer.record("Softmax", "2D Softmax: Row-wise Application",
            "Apply softmax independently to each row (each row is a distribution)",
            {"Input shape: [" + std::to_string(rows) + " × " + std::to_string(cols) + "]",
             "Each row gets its own softmax (rows are independent)",
             "Common in: attention scores, final layer predictions"});

        std::vector<float> result(rows * cols);
        for (int i = 0; i < rows; i++) {
            // Extract row
            std::vector<float> row(logits.begin() + i * cols, logits.begin() + (i + 1) * cols);
            // Compute softmax
            float max_val = *std::max_element(row.begin(), row.end());
            float sum = 0;
            for (int j = 0; j < cols; j++) {
                result[i * cols + j] = std::exp(row[j] - max_val);
                sum += result[i * cols + j];
            }
            for (int j = 0; j < cols; j++) {
                result[i * cols + j] /= sum;
            }
        }

        return result;
    }

private:
    static std::string format_float(float v) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << v;
        return oss.str();
    }
};

} // namespace inferx::nn
