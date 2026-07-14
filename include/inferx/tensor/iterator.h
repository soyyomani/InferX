#pragma once

#include <inferx/tensor/dtype.h>
#include <inferx/tensor/shape.h>
#include <inferx/tensor/stride.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>

namespace inferx::tensor {

template <DType D>
class TensorIterator {
public:
    using value_type = typename DTypeTraits<D>::type;
    using pointer = value_type*;
    using reference = value_type&;
    using const_pointer = const value_type*;
    using const_reference = const value_type&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    // End iterator
    TensorIterator() noexcept = default;

    // Begin iterator
    TensorIterator(pointer base, const Shape& shape, const Stride& stride, bool is_contiguous)
        : base_(base), numel_(static_cast<size_t>(shape.numel())),
          rank_(static_cast<uint8_t>(shape.rank())), is_contiguous_(is_contiguous) {
        for (size_t i = 0; i < rank_; ++i) {
            shape_[i] = shape[i];
            strides_[i] = stride[i];
        }
    }

    // Create end sentinel
    static TensorIterator end_for(pointer base, size_t numel) {
        TensorIterator it;
        it.base_ = base + numel; // for contiguous comparison
        it.flat_idx_ = numel;
        it.numel_ = numel;
        return it;
    }

    TensorIterator& operator++() noexcept {
        ++flat_idx_;
        if (is_contiguous_) {
            ++base_;
        } else {
            // Advance multi-dim index with carry
            for (int i = static_cast<int>(rank_) - 1; i >= 0; --i) {
                ++indices_[i];
                if (indices_[i] < shape_[i]) {
                    base_ += strides_[i];
                    break;
                } else {
                    base_ -= (shape_[i] - 1) * strides_[i];
                    indices_[i] = 0;
                }
            }
        }
        return *this;
    }

    TensorIterator operator++(int) noexcept {
        auto copy = *this;
        ++(*this);
        return copy;
    }

    [[nodiscard]] reference operator*() noexcept { return *base_; }
    [[nodiscard]] pointer operator->() noexcept { return base_; }
    [[nodiscard]] const_reference operator*() const noexcept { return *base_; }
    [[nodiscard]] const_pointer operator->() const noexcept { return base_; }

    bool operator==(const TensorIterator& other) const noexcept {
        return flat_idx_ == other.flat_idx_;
    }
    bool operator!=(const TensorIterator& other) const noexcept {
        return flat_idx_ != other.flat_idx_;
    }

private:
    pointer base_ = nullptr;
    size_t flat_idx_ = 0;
    size_t numel_ = 0;
    std::array<int64_t, Shape::kMaxRank> indices_{};
    std::array<int64_t, Shape::kMaxRank> shape_{};
    std::array<int64_t, Shape::kMaxRank> strides_{};
    uint8_t rank_ = 0;
    bool is_contiguous_ = true;
};

} // namespace inferx::tensor
