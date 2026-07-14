#include <inferx/tensor/stride.h>
#include <inferx/tensor/tracer.h>
#include <string>

using namespace std;

namespace inferx::tensor {

Stride::Stride(span<const int64_t> values) {
    rank_ = static_cast<uint8_t>(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        strides_[i] = values[i];
    }
}

Stride Stride::from_shape(const Shape& shape) noexcept {
    Stride result;
    result.rank_ = static_cast<uint8_t>(shape.rank());
    if (shape.rank() == 0) return result;

    result.strides_[shape.rank() - 1] = 1;
    for (int i = static_cast<int>(shape.rank()) - 2; i >= 0; --i) {
        result.strides_[i] = result.strides_[i + 1] * shape[i + 1];
    }

    auto& tracer = Tracer::instance();
    if (tracer.is_enabled()) {
        vector<string> details;
        details.push_back("row-major (C-contiguous):");
        details.push_back("  strides[" + to_string(shape.rank() - 1) + "] = 1");
        for (int i = static_cast<int>(shape.rank()) - 2; i >= 0; --i) {
            details.push_back("  strides[" + to_string(i) + "] = strides[" +
                to_string(i + 1) + "] x shape[" + to_string(i + 1) + "] = " +
                to_string(result.strides_[i + 1]) + " x " + to_string(shape[i + 1]) +
                " = " + to_string(result.strides_[i]));
        }
        string str;
        for (size_t i = 0; i < shape.rank(); ++i) {
            if (i) str += ", ";
            str += to_string(result.strides_[i]);
        }
        tracer.record("Stride", "from_shape() -> [" + str + "]",
                      "Row-major stride computation", details);
    }

    return result;
}

Stride Stride::from_shape_col_major(const Shape& shape) noexcept {
    Stride result;
    result.rank_ = static_cast<uint8_t>(shape.rank());
    if (shape.rank() == 0) return result;

    result.strides_[0] = 1;
    for (size_t i = 1; i < shape.rank(); ++i) {
        result.strides_[i] = result.strides_[i - 1] * shape[i - 1];
    }
    return result;
}

size_t Stride::offset(span<const int64_t> indices) const noexcept {
    size_t result = 0;
    for (size_t i = 0; i < rank_; ++i) {
        result += static_cast<size_t>(indices[i]) * static_cast<size_t>(strides_[i]);
    }
    return result;
}

bool Stride::is_contiguous(const Shape& shape) const noexcept {
    if (rank_ != shape.rank()) return false;
    Stride expected = from_shape(shape);
    return *this == expected;
}

bool Stride::operator==(const Stride& other) const noexcept {
    if (rank_ != other.rank_) return false;
    for (size_t i = 0; i < rank_; ++i) {
        if (strides_[i] != other.strides_[i]) return false;
    }
    return true;
}

} // namespace inferx::tensor
