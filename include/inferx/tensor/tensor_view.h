#pragma once

#include <inferx/tensor/dtype.h>
#include <inferx/tensor/shape.h>
#include <inferx/tensor/stride.h>
#include <inferx/tensor/tensor.h>

#include <cassert>
#include <array>
#include <span>
#include <cstdint>
#include <algorithm>

namespace inferx::tensor {

template <DType D>
class TensorView {
public:
    using value_type = typename DTypeTraits<D>::type;
    using const_pointer = const value_type*;
    using const_reference = const value_type&;

    // Implicit conversion from Tensor
    TensorView(const Tensor<D>& tensor) noexcept
        : data_(tensor.data()), shape_(tensor.shape()), stride_(tensor.stride()) {}

    // Accessors
    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] const Stride& stride() const noexcept { return stride_; }
    [[nodiscard]] size_t rank() const noexcept { return shape_.rank(); }
    [[nodiscard]] int64_t numel() const noexcept { return shape_.numel(); }
    [[nodiscard]] bool is_contiguous() const noexcept { return stride_.is_contiguous(shape_); }
    [[nodiscard]] const_pointer data() const noexcept { return data_; }

    // Read-only element access
    template <typename... Indices>
        requires (sizeof...(Indices) > 0) && (std::is_integral_v<Indices> && ...)
    [[nodiscard]] const_reference operator()(Indices... indices) const {
        const std::array<int64_t, sizeof...(Indices)> idx = {static_cast<int64_t>(indices)...};
        #ifndef NDEBUG
        for (size_t i = 0; i < idx.size(); ++i) {
            assert(idx[i] >= 0 && idx[i] < shape_[i]);
        }
        #endif
        size_t off = stride_.offset(std::span<const int64_t>(idx.data(), idx.size()));
        return *(data_ + off);
    }

    [[nodiscard]] const_reference flat(size_t idx) const {
        assert(idx < static_cast<size_t>(numel()));
        return *(data_ + idx);
    }

    // Slice returns a new view
    [[nodiscard]] TensorView slice(size_t dim, int64_t start, int64_t end) const {
        assert(dim < rank());
        int64_t dim_size = shape_[dim];
        if (start < 0) start += dim_size;
        if (end < 0) end += dim_size;
        start = std::clamp(start, int64_t(0), dim_size);
        end = std::clamp(end, start, dim_size);

        std::array<int64_t, Shape::kMaxRank> new_dims{};
        for (size_t i = 0; i < rank(); ++i) {
            new_dims[i] = (i == dim) ? (end - start) : shape_[i];
        }

        TensorView result;
        result.data_ = data_ + start * stride_[dim];
        result.shape_ = Shape(std::span<const int64_t>(new_dims.data(), rank()));
        result.stride_ = stride_;
        return result;
    }

private:
    TensorView() = default;

    const_pointer data_ = nullptr;
    Shape shape_;
    Stride stride_;
};

} // namespace inferx::tensor
