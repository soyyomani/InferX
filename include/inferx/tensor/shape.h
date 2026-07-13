#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>

namespace inferx::tensor {

class Shape {
public:
    static constexpr size_t kMaxRank = 8;

    // Construction
    Shape() noexcept = default;
    explicit Shape(std::initializer_list<int64_t> dims);
    explicit Shape(std::span<const int64_t> dims);

    // Accessors
    [[nodiscard]] size_t rank() const noexcept { return rank_; }
    [[nodiscard]] int64_t operator[](size_t idx) const noexcept { return dims_[idx]; }
    [[nodiscard]] int64_t at(size_t idx) const;
    [[nodiscard]] int64_t numel() const noexcept;
    [[nodiscard]] bool is_scalar() const noexcept { return rank_ == 0; }

    // Shape operations
    [[nodiscard]] Shape squeeze(size_t dim) const;
    [[nodiscard]] Shape unsqueeze(size_t dim) const;
    [[nodiscard]] Shape permute(std::span<const size_t> order) const;

    // Broadcasting
    [[nodiscard]] static std::optional<Shape> broadcast(const Shape& a, const Shape& b);
    [[nodiscard]] bool is_broadcastable_with(const Shape& other) const noexcept;

    // Comparison
    [[nodiscard]] bool operator==(const Shape& other) const noexcept;
    [[nodiscard]] bool operator!=(const Shape& other) const noexcept { return !(*this == other); }

    // Iteration
    [[nodiscard]] auto begin() const noexcept { return dims_.begin(); }
    [[nodiscard]] auto end() const noexcept { return dims_.begin() + rank_; }

private:
    std::array<int64_t, kMaxRank> dims_{};
    uint8_t rank_ = 0;

    static void validate(std::span<const int64_t> dims);
};

} // namespace inferx::tensor
