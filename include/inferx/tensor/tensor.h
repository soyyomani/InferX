#pragma once

#include <inferx/tensor/dtype.h>
#include <inferx/tensor/shape.h>
#include <inferx/tensor/stride.h>
#include <inferx/tensor/storage.h>

#include <cassert>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <cstring>
#include <algorithm>

namespace inferx::tensor {

template <DType D>
class Tensor {
public:
    using value_type = typename DTypeTraits<D>::type;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;

    // Construction
    explicit Tensor(Shape shape)
        : storage_(std::make_shared<TensorStorage>(
              static_cast<size_t>(shape.numel()) * DTypeTraits<D>::size)),
          shape_(shape),
          stride_(Stride::from_shape(shape)) {}

    Tensor(Shape shape, value_type fill_value) : Tensor(shape) {
        fill_(fill_value);
    }

    Tensor(Shape shape, std::span<const value_type> data) : Tensor(shape) {
        if (static_cast<int64_t>(data.size()) != shape.numel()) {
            throw std::invalid_argument(
                "Tensor: span size " + std::to_string(data.size()) +
                " != shape numel " + std::to_string(shape.numel()));
        }
        std::memcpy(this->data(), data.data(), data.size() * sizeof(value_type));
    }

    static Tensor from_external(Shape shape, pointer data, size_t /*size_bytes*/) {
        Tensor t;
        t.shape_ = shape;
        t.stride_ = Stride::from_shape(shape);
        t.external_data_ = data;
        return t;
    }

    // Named constructors
    [[nodiscard]] static Tensor zeros(Shape shape) { return Tensor(shape); }
    [[nodiscard]] static Tensor ones(Shape shape) { return Tensor(shape, value_type(1)); }
    [[nodiscard]] static Tensor full(Shape shape, value_type value) { return Tensor(shape, value); }

    // Accessors
    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] const Stride& stride() const noexcept { return stride_; }
    [[nodiscard]] constexpr DType dtype() const noexcept { return D; }
    [[nodiscard]] size_t rank() const noexcept { return shape_.rank(); }
    [[nodiscard]] int64_t numel() const noexcept { return shape_.numel(); }
    [[nodiscard]] size_t size_bytes() const noexcept { return static_cast<size_t>(numel()) * DTypeTraits<D>::size; }
    [[nodiscard]] bool is_contiguous() const noexcept { return stride_.is_contiguous(shape_); }

    [[nodiscard]] pointer data() noexcept {
        if (external_data_) return external_data_;
        return reinterpret_cast<pointer>(
            static_cast<uint8_t*>(storage_->data()) + offset_);
    }
    [[nodiscard]] const_pointer data() const noexcept {
        if (external_data_) return external_data_;
        return reinterpret_cast<const_pointer>(
            static_cast<const uint8_t*>(storage_->data()) + offset_);
    }

    // Element access: tensor(i, j, k)
    template <typename... Indices>
        requires (sizeof...(Indices) > 0) && (std::is_integral_v<Indices> && ...)
    [[nodiscard]] reference operator()(Indices... indices) {
        static_assert(sizeof...(Indices) <= Shape::kMaxRank);
        const std::array<int64_t, sizeof...(Indices)> idx = {static_cast<int64_t>(indices)...};
        #ifndef NDEBUG
        for (size_t i = 0; i < idx.size(); ++i) {
            assert(idx[i] >= 0 && idx[i] < shape_[i]);
        }
        #endif
        size_t off = stride_.offset(std::span<const int64_t>(idx.data(), idx.size()));
        return *(data() + off);
    }

    template <typename... Indices>
        requires (sizeof...(Indices) > 0) && (std::is_integral_v<Indices> && ...)
    [[nodiscard]] const_reference operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) <= Shape::kMaxRank);
        const std::array<int64_t, sizeof...(Indices)> idx = {static_cast<int64_t>(indices)...};
        #ifndef NDEBUG
        for (size_t i = 0; i < idx.size(); ++i) {
            assert(idx[i] >= 0 && idx[i] < shape_[i]);
        }
        #endif
        size_t off = stride_.offset(std::span<const int64_t>(idx.data(), idx.size()));
        return *(data() + off);
    }

    // Flat access
    [[nodiscard]] reference flat(size_t idx) {
        assert(idx < static_cast<size_t>(numel()));
        return *(data() + idx);
    }
    [[nodiscard]] const_reference flat(size_t idx) const {
        assert(idx < static_cast<size_t>(numel()));
        return *(data() + idx);
    }

    // View operations (zero-copy)
    [[nodiscard]] Tensor reshape(Shape new_shape) const {
        if (new_shape.numel() != shape_.numel()) {
            throw std::invalid_argument(
                "reshape: element count mismatch (" +
                std::to_string(shape_.numel()) + " vs " +
                std::to_string(new_shape.numel()) + ")");
        }
        if (is_contiguous()) {
            Tensor result;
            result.storage_ = storage_;
            result.shape_ = new_shape;
            result.stride_ = Stride::from_shape(new_shape);
            result.offset_ = offset_;
            return result;
        }
        // Non-contiguous: copy first
        Tensor contig = contiguous();
        return contig.reshape(new_shape);
    }

    [[nodiscard]] Tensor slice(size_t dim, int64_t start, int64_t end) const {
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

        Tensor result;
        result.storage_ = storage_;
        result.shape_ = Shape(std::span<const int64_t>(new_dims.data(), rank()));
        result.stride_ = stride_;
        result.offset_ = offset_ + static_cast<size_t>(start) *
            static_cast<size_t>(stride_[dim]) * sizeof(value_type);
        return result;
    }

    [[nodiscard]] Tensor transpose(size_t dim0, size_t dim1) const {
        assert(dim0 < rank() && dim1 < rank());
        std::array<int64_t, Shape::kMaxRank> new_dims{};
        std::array<int64_t, Shape::kMaxRank> new_strides{};
        for (size_t i = 0; i < rank(); ++i) {
            new_dims[i] = shape_[i];
            new_strides[i] = stride_[i];
        }
        std::swap(new_dims[dim0], new_dims[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);

        Tensor result;
        result.storage_ = storage_;
        result.shape_ = Shape(std::span<const int64_t>(new_dims.data(), rank()));
        result.stride_ = Stride(std::span<const int64_t>(new_strides.data(), rank()));
        result.offset_ = offset_;
        return result;
    }

    [[nodiscard]] Tensor contiguous() const {
        if (is_contiguous()) {
            return *this; // shared storage
        }
        Tensor result(shape_);
        // Copy in logical order
        for (int64_t i = 0; i < numel(); ++i) {
            // Compute multi-dim index from flat
            std::array<int64_t, Shape::kMaxRank> idx{};
            int64_t remaining = i;
            Stride row_stride = Stride::from_shape(shape_);
            for (size_t d = 0; d < rank(); ++d) {
                idx[d] = remaining / row_stride[d];
                remaining -= idx[d] * row_stride[d];
            }
            size_t src_off = stride_.offset(std::span<const int64_t>(idx.data(), rank()));
            result.flat(static_cast<size_t>(i)) = *(data() + src_off);
        }
        return result;
    }

    // In-place operations
    Tensor& fill_(value_type value) noexcept {
        pointer p = data();
        for (int64_t i = 0; i < numel(); ++i) {
            p[i] = value;
        }
        return *this;
    }

    Tensor& zero_() noexcept {
        return fill_(value_type(0));
    }

    // Deep copy
    [[nodiscard]] Tensor clone() const {
        Tensor result(shape_);
        if (is_contiguous()) {
            std::memcpy(result.data(), data(), size_bytes());
        } else {
            for (int64_t i = 0; i < numel(); ++i) {
                std::array<int64_t, Shape::kMaxRank> idx{};
                int64_t remaining = i;
                Stride row_stride = Stride::from_shape(shape_);
                for (size_t d = 0; d < rank(); ++d) {
                    idx[d] = remaining / row_stride[d];
                    remaining -= idx[d] * row_stride[d];
                }
                size_t src_off = stride_.offset(std::span<const int64_t>(idx.data(), rank()));
                result.flat(static_cast<size_t>(i)) = *(data() + src_off);
            }
        }
        return result;
    }

    // Copy/move: shared storage semantics
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

private:
    Tensor() = default; // for internal factory use

    std::shared_ptr<TensorStorage> storage_;
    Shape shape_;
    Stride stride_;
    size_t offset_ = 0;
    pointer external_data_ = nullptr;
};

} // namespace inferx::tensor
