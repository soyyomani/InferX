#include <inferx/tensor/shape.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

using namespace std;

namespace inferx::tensor {

// Validation

void Shape::validate(span<const int64_t> dims) {
    if (dims.size() > kMaxRank) {
        throw invalid_argument(
            "Shape rank " + to_string(dims.size()) +
            " exceeds maximum supported rank of " + to_string(kMaxRank));
    }
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] <= 0) {
            throw invalid_argument(
                "Shape dimension " + to_string(i) +
                " must be positive, got " + to_string(dims[i]));
        }
    }
}

// Construction

Shape::Shape(initializer_list<int64_t> dims)
    : Shape(span<const int64_t>(dims.begin(), dims.size())) {}

Shape::Shape(span<const int64_t> dims) {
    validate(dims);
    rank_ = static_cast<uint8_t>(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        dims_[i] = dims[i];
    }
}

// Accessors

int64_t Shape::at(size_t idx) const {
    if (idx >= rank_) {
        throw out_of_range(
            "Shape index " + to_string(idx) +
            " out of range for rank " + to_string(rank_));
    }
    return dims_[idx];
}

int64_t Shape::numel() const noexcept {
    if (rank_ == 0) return 1;
    int64_t total = 1;
    for (size_t i = 0; i < rank_; ++i) {
        total *= dims_[i];
    }
    return total;
}

// Shape operations

Shape Shape::squeeze(size_t dim) const {
    if (dim >= rank_) {
        throw invalid_argument(
            "squeeze: dimension " + to_string(dim) +
            " out of range for rank " + to_string(rank_));
    }
    if (dims_[dim] != 1) {
        throw invalid_argument(
            "squeeze: dimension " + to_string(dim) +
            " has size " + to_string(dims_[dim]) + ", expected 1");
    }

    array<int64_t, kMaxRank> new_dims{};
    size_t new_rank = 0;
    for (size_t i = 0; i < rank_; ++i) {
        if (i != dim) {
            new_dims[new_rank++] = dims_[i];
        }
    }

    return Shape(span<const int64_t>(new_dims.data(), new_rank));
}

Shape Shape::unsqueeze(size_t dim) const {
    if (dim > rank_) {
        throw invalid_argument(
            "unsqueeze: dimension " + to_string(dim) +
            " out of range for rank " + to_string(rank_) +
            " (must be <= rank)");
    }
    if (rank_ + 1 > kMaxRank) {
        throw invalid_argument(
            "unsqueeze: resulting rank " + to_string(rank_ + 1) +
            " exceeds maximum supported rank of " + to_string(kMaxRank));
    }

    array<int64_t, kMaxRank> new_dims{};
    for (size_t i = 0; i < dim; ++i) {
        new_dims[i] = dims_[i];
    }
    new_dims[dim] = 1;
    for (size_t i = dim; i < rank_; ++i) {
        new_dims[i + 1] = dims_[i];
    }

    return Shape(span<const int64_t>(new_dims.data(), rank_ + 1));
}

Shape Shape::permute(span<const size_t> order) const {
    if (order.size() != rank_) {
        throw invalid_argument(
            "permute: order size " + to_string(order.size()) +
            " does not match rank " + to_string(rank_));
    }

    array<bool, kMaxRank> seen{};
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] >= rank_) {
            throw invalid_argument(
                "permute: index " + to_string(order[i]) +
                " out of range for rank " + to_string(rank_));
        }
        if (seen[order[i]]) {
            throw invalid_argument(
                "permute: duplicate index " + to_string(order[i]) +
                " in permutation order");
        }
        seen[order[i]] = true;
    }

    array<int64_t, kMaxRank> new_dims{};
    for (size_t i = 0; i < rank_; ++i) {
        new_dims[i] = dims_[order[i]];
    }

    return Shape(span<const int64_t>(new_dims.data(), rank_));
}

// Broadcasting

optional<Shape> Shape::broadcast(const Shape& a, const Shape& b) {
    const size_t out_rank = max(a.rank(), b.rank());

    if (out_rank > kMaxRank) {
        return nullopt;
    }

    array<int64_t, kMaxRank> out_dims{};

    for (size_t i = 0; i < out_rank; ++i) {
        const int a_idx = static_cast<int>(a.rank()) - static_cast<int>(out_rank) + static_cast<int>(i);
        const int b_idx = static_cast<int>(b.rank()) - static_cast<int>(out_rank) + static_cast<int>(i);

        const int64_t dim_a = (a_idx >= 0) ? a[static_cast<size_t>(a_idx)] : 1;
        const int64_t dim_b = (b_idx >= 0) ? b[static_cast<size_t>(b_idx)] : 1;

        if (dim_a == dim_b) {
            out_dims[i] = dim_a;
        } else if (dim_a == 1) {
            out_dims[i] = dim_b;
        } else if (dim_b == 1) {
            out_dims[i] = dim_a;
        } else {
            return nullopt;
        }
    }

    return Shape(span<const int64_t>(out_dims.data(), out_rank));
}

bool Shape::is_broadcastable_with(const Shape& other) const noexcept {
    return broadcast(*this, other).has_value();
}

// Comparison

bool Shape::operator==(const Shape& other) const noexcept {
    if (rank_ != other.rank_) return false;
    for (size_t i = 0; i < rank_; ++i) {
        if (dims_[i] != other.dims_[i]) return false;
    }
    return true;
}

} // namespace inferx::tensor
