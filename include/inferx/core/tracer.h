#pragma once

#include <string>
#include <vector>

namespace inferx::core {

struct TraceStep {
    std::string component;
    std::string title;
    std::string detail;
    std::vector<std::string> internal;
};

/// Global tracer that records step-by-step execution for visualization.
/// Each module (tokenizer, embedding, attention, etc.) records its math steps here.
class Tracer {
public:
    static Tracer& instance() {
        static Tracer t;
        return t;
    }

    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool is_enabled() const { return enabled_; }

    void clear() { steps_.clear(); }
    const std::vector<TraceStep>& steps() const { return steps_; }

    std::vector<TraceStep> take() {
        auto result = std::move(steps_);
        steps_.clear();
        return result;
    }

    void record(const std::string& component,
                const std::string& title,
                const std::string& detail,
                const std::vector<std::string>& internal = {}) {
        if (!enabled_) return;
        steps_.push_back({component, title, detail, internal});
    }

private:
    Tracer() = default;
    bool enabled_ = false;
    std::vector<TraceStep> steps_;
};

} // namespace inferx::core
