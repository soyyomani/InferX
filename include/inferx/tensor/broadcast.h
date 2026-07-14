#pragma once

#include <inferx/tensor/shape.h>
#include <inferx/tensor/stride.h>

#include <optional>
#include <string>
#include <variant>

namespace inferx::tensor {

class BroadcastEngine {
public:
    struct BroadcastResult {
        Shape output_shape;
        Stride stride_a;
        Stride stride_b;
        bool a_needs_broadcast;
        bool b_needs_broadcast;
    };

    // Returns BroadcastResult on success, error string on failure
    using ComputeResult = std::variant<BroadcastResult, std::string>;

    // Compute broadcast result for two shapes
    [[nodiscard]] static ComputeResult compute(const Shape& a, const Shape& b);

    // Check compatibility only
    [[nodiscard]] static bool compatible(const Shape& a, const Shape& b) noexcept;

    // Compute output shape only
    [[nodiscard]] static std::optional<Shape>
    output_shape(const Shape& a, const Shape& b) noexcept;
};

} // namespace inferx::tensor
