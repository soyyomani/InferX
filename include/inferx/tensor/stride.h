#pragma once

#include <inferx/tensor/shape.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace inferx::tensor {

class Stride {
public:
    Stride() noexcept = default;

    // Construct directly from values
    explicit Stride(std::span<const int64_t> values);

    // Compute row-major (C-contiguous) strides from shape
    [[nodiscard]] static Stride from_shape(const Shape& shape) noexcept;

    // Compute column-major (Fortran) strides from shape
    [[nodiscard]] static Stride from_shape_col_major(const Shape& shape) noexcept;

    // Accessors
    [[nodiscard]] int64_t operator[](size_t dim) const noexcept { return strides_[dim]; }
    [[nodiscard]] size_t rank() const noexcept { return rank_; }

    // Compute flat offset from multi-dimensional index
    [[nodiscard]] size_t offset(std::span<const int64_t> indices) const noexcept;

    // Check if these strides represent contiguous layout for given shape
    [[nodiscard]] bool is_contiguous(const Shape& shape) const noexcept;

    [[nodiscard]] bool operator==(const Stride& other) const noexcept;
    [[nodiscard]] bool operator!=(const Stride& other) const noexcept { return !(*this == other); }

private:
    std::array<int64_t, Shape::kMaxRank> strides_{};
    uint8_t rank_ = 0;
};

} // namespace inferx::tensor
