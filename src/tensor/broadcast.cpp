#include <inferx/tensor/broadcast.h>
#include <inferx/tensor/tracer.h>

#include <algorithm>
#include <string>

using namespace std;

namespace inferx::tensor {

BroadcastEngine::ComputeResult
BroadcastEngine::compute(const Shape& a, const Shape& b) {
    const size_t out_rank = max(a.rank(), b.rank());

    if (out_rank > Shape::kMaxRank) {
        return string("Broadcast: output rank " + to_string(out_rank) + " exceeds max");
    }

    array<int64_t, Shape::kMaxRank> out_dims{};
    array<int64_t, Shape::kMaxRank> stride_a{};
    array<int64_t, Shape::kMaxRank> stride_b{};

    int64_t a_acc = 1;
    int64_t b_acc = 1;

    for (int i = static_cast<int>(out_rank) - 1; i >= 0; --i) {
        int a_idx = static_cast<int>(a.rank()) - static_cast<int>(out_rank) + i;
        int b_idx = static_cast<int>(b.rank()) - static_cast<int>(out_rank) + i;

        int64_t dim_a = (a_idx >= 0) ? a[static_cast<size_t>(a_idx)] : 1;
        int64_t dim_b = (b_idx >= 0) ? b[static_cast<size_t>(b_idx)] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return string("Shapes not broadcastable: dim " + to_string(i) +
                " has " + to_string(dim_a) + " vs " + to_string(dim_b));
        }

        out_dims[i] = max(dim_a, dim_b);
        stride_a[i] = (dim_a == 1 && out_dims[i] > 1) ? 0 : a_acc;
        stride_b[i] = (dim_b == 1 && out_dims[i] > 1) ? 0 : b_acc;

        if (dim_a > 1) a_acc *= dim_a;
        if (dim_b > 1) b_acc *= dim_b;
    }

    Shape out_shape(span<const int64_t>(out_dims.data(), out_rank));
    Stride sa(span<const int64_t>(stride_a.data(), out_rank));
    Stride sb(span<const int64_t>(stride_b.data(), out_rank));

    BroadcastResult result{
        out_shape, sa, sb,
        a.numel() < out_shape.numel(),
        b.numel() < out_shape.numel()
    };

    auto& tracer = Tracer::instance();
    if (tracer.is_enabled()) {
        vector<string> details;
        string a_str, b_str, out_str;
        for (size_t i = 0; i < a.rank(); ++i) { if (i) a_str += ","; a_str += to_string(a[i]); }
        for (size_t i = 0; i < b.rank(); ++i) { if (i) b_str += ","; b_str += to_string(b[i]); }
        for (size_t i = 0; i < out_shape.rank(); ++i) { if (i) out_str += ","; out_str += to_string(out_shape[i]); }

        details.push_back("A=[" + a_str + "], B=[" + b_str + "]");
        details.push_back("output rank = " + to_string(out_rank));
        for (size_t i = 0; i < out_rank; ++i) {
            int a_idx2 = static_cast<int>(a.rank()) - static_cast<int>(out_rank) + static_cast<int>(i);
            int b_idx2 = static_cast<int>(b.rank()) - static_cast<int>(out_rank) + static_cast<int>(i);
            int64_t da = (a_idx2 >= 0) ? a[a_idx2] : 1;
            int64_t db = (b_idx2 >= 0) ? b[b_idx2] : 1;
            details.push_back("  dim " + to_string(i) + ": A=" + to_string(da) +
                " B=" + to_string(db) + " -> " + to_string(out_dims[i]) +
                (stride_a[i] == 0 ? " (A broadcast)" : "") +
                (stride_b[i] == 0 ? " (B broadcast)" : ""));
        }
        details.push_back("output: [" + out_str + "]");

        tracer.record("BroadcastEngine", "compute([" + a_str + "], [" + b_str + "])",
                      "Compatible -> [" + out_str + "]", details);
    }

    return result;
}

bool BroadcastEngine::compatible(const Shape& a, const Shape& b) noexcept {
    return Shape::broadcast(a, b).has_value();
}

optional<Shape> BroadcastEngine::output_shape(const Shape& a, const Shape& b) noexcept {
    return Shape::broadcast(a, b);
}

} // namespace inferx::tensor
