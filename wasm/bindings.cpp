#include <emscripten/bind.h>
#include <inferx/tensor/dtype.h>
#include <inferx/tensor/shape.h>
#include <inferx/tensor/stride.h>
#include <inferx/tensor/storage.h>
#include <inferx/tensor/tensor.h>
#include <inferx/tensor/broadcast.h>
#include <inferx/tensor/iterator.h>
#include <inferx/tensor/tracer.h>

#include <string>
#include <vector>

using namespace std;
using namespace emscripten;
using namespace inferx::tensor;

// JS-friendly trace step
struct JSTraceStep {
    string component;
    string title;
    string detail;
    vector<string> internal;
};

// Helper to flush tracer into result vector
vector<JSTraceStep> flushTracer() {
    auto& tracer = Tracer::instance();
    vector<JSTraceStep> result;
    for (auto& step : tracer.steps()) {
        result.push_back({step.component, step.title, step.detail, step.internal});
    }
    return result;
}

// Full pipeline: create tensor with all steps visible
vector<JSTraceStep> traceFullCreate(string dtype_name, vector<int64_t> dims, string fill_mode) {
    auto& tracer = Tracer::instance();
    tracer.enable();
    tracer.clear();

    try {
        // Step 1: DType resolution
        size_t elem_size = 4, elem_align = 4;
        string cpp_type = "float";
        if (dtype_name == "Float16") { elem_size = 2; elem_align = 2; cpp_type = "uint16_t"; }
        else if (dtype_name == "Int8") { elem_size = 1; elem_align = 1; cpp_type = "int8_t"; }
        else if (dtype_name == "Int32") { elem_size = 4; elem_align = 4; cpp_type = "int32_t"; }
        else if (dtype_name == "Int64") { elem_size = 8; elem_align = 8; cpp_type = "int64_t"; }
        else if (dtype_name == "Bool") { elem_size = 1; elem_align = 1; cpp_type = "uint8_t"; }

        tracer.record("DType", "Resolve DType::" + dtype_name,
            "size=" + to_string(elem_size) + "B, align=" + to_string(elem_align) + "B, type=" + cpp_type,
            {
                "template match: DType::" + dtype_name,
                "DTypeTraits<DType::" + dtype_name + ">::type = " + cpp_type,
                "DTypeTraits<DType::" + dtype_name + ">::size = " + to_string(elem_size),
                "DTypeTraits<DType::" + dtype_name + ">::alignment = " + to_string(elem_align),
                "TensorScalar<" + cpp_type + "> satisfied: true",
            });

        // Step 2: Shape construction (tracer called inside Shape constructor)
        auto sp = span<const int64_t>(dims.data(), dims.size());
        Shape shape(sp);

        // Step 3: Stride (tracer called inside from_shape)
        Stride stride = Stride::from_shape(shape);

        // Step 4: Storage allocation (tracer called inside TensorStorage)
        size_t total_bytes = static_cast<size_t>(shape.numel()) * elem_size;
        TensorStorage storage(total_bytes);

        // Step 5: Tensor assembly
        string dim_str, stride_str;
        for (size_t i = 0; i < shape.rank(); ++i) {
            if (i) { dim_str += ", "; stride_str += ", "; }
            dim_str += to_string(shape[i]);
            stride_str += to_string(stride[i]);
        }

        vector<string> tensor_details = {
            "storage_ = shared_ptr<TensorStorage> (refcount=1)",
            "shape_ = [" + dim_str + "]",
            "stride_ = [" + stride_str + "]",
            "offset_ = 0",
            "is_contiguous() = true",
            "numel() = " + to_string(shape.numel()),
            "size_bytes() = " + to_string(total_bytes),
            "data() -> aligned pointer (16B boundary)",
        };

        if (fill_mode == "zeros") {
            tensor_details.push_back("fill: already zero (storage is zero-initialized)");
        } else if (fill_mode == "ones") {
            tensor_details.push_back("fill_(1.0): write 1.0 to all " + to_string(shape.numel()) + " elements");
        }

        tracer.record("Tensor", "Tensor<" + dtype_name + "> created",
            fill_mode + "([" + dim_str + "])", tensor_details);

    } catch (const exception& e) {
        tracer.record("Error", "Exception", e.what(), {e.what()});
    }

    auto result = flushTracer();
    tracer.disable();
    return result;
}

// Trace element access
vector<JSTraceStep> traceAccess(vector<int64_t> dims, vector<int64_t> indices) {
    auto& tracer = Tracer::instance();
    tracer.enable();
    tracer.clear();

    try {
        Shape shape(span<const int64_t>(dims.data(), dims.size()));
        Stride stride = Stride::from_shape(shape);

        vector<string> details;
        string idx_str;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (i) idx_str += ", ";
            idx_str += to_string(indices[i]);
        }

        details.push_back("static_assert: " + to_string(indices.size()) + " indices == rank " +
            to_string(shape.rank()) + " ok");
        details.push_back("");
        details.push_back("#ifndef NDEBUG // bounds check:");
        for (size_t i = 0; i < indices.size(); ++i) {
            bool ok = indices[i] >= 0 && indices[i] < shape[i];
            details.push_back("  assert(" + to_string(indices[i]) + " >= 0 && " +
                to_string(indices[i]) + " < " + to_string(shape[i]) + ") " +
                (ok ? "ok" : "FAIL"));
        }
        details.push_back("#endif");
        details.push_back("");

        details.push_back("offset = sum(index[i] * stride[i]):");
        int64_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            int64_t contrib = indices[i] * stride[i];
            details.push_back("  " + to_string(indices[i]) + " * " + to_string(stride[i]) +
                " = " + to_string(contrib));
            offset += contrib;
        }
        details.push_back("  flat offset = " + to_string(offset));
        details.push_back("");
        details.push_back("return *(data() + " + to_string(offset) + ")");
        details.push_back("byte address = base + " + to_string(offset * 4));

        tracer.record("Tensor::operator()", "t(" + idx_str + ") -> offset " + to_string(offset),
            "Stride multiplication: " + to_string(offset) + " elements from base", details);

    } catch (const exception& e) {
        tracer.record("Error", "Exception", e.what(), {e.what()});
    }

    auto result = flushTracer();
    tracer.disable();
    return result;
}

// Trace reshape
vector<JSTraceStep> traceReshape(vector<int64_t> old_dims, vector<int64_t> new_dims) {
    auto& tracer = Tracer::instance();
    tracer.enable();
    tracer.clear();

    try {
        Shape old_shape(span<const int64_t>(old_dims.data(), old_dims.size()));
        Shape new_shape(span<const int64_t>(new_dims.data(), new_dims.size()));
        Stride new_stride = Stride::from_shape(new_shape);

        string old_str, new_str, new_stride_str;
        for (size_t i = 0; i < old_shape.rank(); ++i) { if (i) old_str += ","; old_str += to_string(old_shape[i]); }
        for (size_t i = 0; i < new_shape.rank(); ++i) { if (i) new_str += ","; new_str += to_string(new_shape[i]); }
        for (size_t i = 0; i < new_shape.rank(); ++i) { if (i) new_stride_str += ","; new_stride_str += to_string(new_stride[i]); }

        vector<string> details;
        details.push_back("old shape: [" + old_str + "], numel = " + to_string(old_shape.numel()));
        details.push_back("new shape: [" + new_str + "], numel = " + to_string(new_shape.numel()));

        if (old_shape.numel() != new_shape.numel()) {
            details.push_back(to_string(old_shape.numel()) + " != " + to_string(new_shape.numel()) +
                " -> throw std::invalid_argument");
            tracer.record("Tensor::reshape", "reshape([" + new_str + "]) FAILED",
                "Element count mismatch", details);
        } else {
            details.push_back(to_string(old_shape.numel()) + " == " + to_string(new_shape.numel()) + " ok");
            details.push_back("");
            details.push_back("is_contiguous() = true -> zero-copy path");
            details.push_back("");
            details.push_back("result.storage_ = this->storage_ (shared_ptr copy, refcount++)");
            details.push_back("result.shape_ = [" + new_str + "]");
            details.push_back("result.stride_ = [" + new_stride_str + "]");
            details.push_back("result.offset_ = unchanged");
            details.push_back("");
            details.push_back("NO memory allocation. NO data copy.");

            tracer.record("Tensor::reshape", "reshape([" + new_str + "]) -> zero-copy",
                "Metadata-only change, same storage", details);
        }
    } catch (const exception& e) {
        tracer.record("Error", "Exception", e.what(), {e.what()});
    }

    auto result = flushTracer();
    tracer.disable();
    return result;
}

// Trace slice
vector<JSTraceStep> traceSlice(vector<int64_t> dims, int dim, int64_t start, int64_t end) {
    auto& tracer = Tracer::instance();
    tracer.enable();
    tracer.clear();

    try {
        Shape shape(span<const int64_t>(dims.data(), dims.size()));
        Stride stride = Stride::from_shape(shape);

        int64_t dim_size = shape[dim];
        int64_t cstart = start < 0 ? start + dim_size : start;
        int64_t cend = end < 0 ? end + dim_size : end;
        cstart = max(int64_t(0), min(cstart, dim_size));
        cend = max(cstart, min(cend, dim_size));
        int64_t slice_size = cend - cstart;
        int64_t offset_elements = cstart * stride[dim];

        string dim_str;
        for (size_t i = 0; i < shape.rank(); ++i) { if (i) dim_str += ","; dim_str += to_string(shape[i]); }

        vector<string> new_dims_arr;
        string new_shape_str;
        for (size_t i = 0; i < shape.rank(); ++i) {
            int64_t d = (static_cast<int>(i) == dim) ? slice_size : shape[i];
            if (i) new_shape_str += ",";
            new_shape_str += to_string(d);
        }

        string stride_str;
        for (size_t i = 0; i < stride.rank(); ++i) { if (i) stride_str += ","; stride_str += to_string(stride[i]); }

        vector<string> details = {
            "input shape: [" + dim_str + "]",
            "dim=" + to_string(dim) + ", start=" + to_string(start) + ", end=" + to_string(end),
            "",
            "normalize negative indices:",
            "  start: " + to_string(start) + " -> " + to_string(cstart),
            "  end: " + to_string(end) + " -> " + to_string(cend),
            "clamp to [0, shape[" + to_string(dim) + "]=" + to_string(dim_size) + "]:",
            "  clamped_start = " + to_string(cstart),
            "  clamped_end = " + to_string(cend),
            "",
            "slice_size = " + to_string(cend) + " - " + to_string(cstart) + " = " + to_string(slice_size),
            "offset_advance = " + to_string(cstart) + " * stride[" + to_string(dim) + "]=" +
                to_string(stride[dim]) + " = " + to_string(offset_elements) + " elements",
            "offset_bytes = " + to_string(offset_elements) + " * 4 = " + to_string(offset_elements * 4),
            "",
            "result.storage_ = same (shared_ptr, refcount++)",
            "result.shape_ = [" + new_shape_str + "]",
            "result.stride_ = [" + stride_str + "] (unchanged)",
            "result.offset_ += " + to_string(offset_elements * 4) + " bytes",
            "",
            "NO allocation. NO copy. View into same buffer.",
        };

        tracer.record("Tensor::slice", "slice(dim=" + to_string(dim) + ", " +
            to_string(start) + ":" + to_string(end) + ") -> [" + new_shape_str + "]",
            "Zero-copy view, offset advance " + to_string(offset_elements * 4) + " bytes", details);

    } catch (const exception& e) {
        tracer.record("Error", "Exception", e.what(), {e.what()});
    }

    auto result = flushTracer();
    tracer.disable();
    return result;
}

// Trace transpose
vector<JSTraceStep> traceTranspose(vector<int64_t> dims, int dim0, int dim1) {
    auto& tracer = Tracer::instance();
    tracer.enable();
    tracer.clear();

    try {
        Shape shape(span<const int64_t>(dims.data(), dims.size()));
        Stride stride = Stride::from_shape(shape);

        string old_shape_str, old_stride_str;
        for (size_t i = 0; i < shape.rank(); ++i) {
            if (i) { old_shape_str += ","; old_stride_str += ","; }
            old_shape_str += to_string(shape[i]);
            old_stride_str += to_string(stride[i]);
        }

        // Compute transposed
        vector<int64_t> new_dims(shape.rank()), new_strides(shape.rank());
        for (size_t i = 0; i < shape.rank(); ++i) {
            new_dims[i] = shape[i];
            new_strides[i] = stride[i];
        }
        swap(new_dims[dim0], new_dims[dim1]);
        swap(new_strides[dim0], new_strides[dim1]);

        string new_shape_str, new_stride_str;
        for (size_t i = 0; i < shape.rank(); ++i) {
            if (i) { new_shape_str += ","; new_stride_str += ","; }
            new_shape_str += to_string(new_dims[i]);
            new_stride_str += to_string(new_strides[i]);
        }

        vector<string> details = {
            "assert(" + to_string(dim0) + " < rank=" + to_string(shape.rank()) + ") ok",
            "assert(" + to_string(dim1) + " < rank=" + to_string(shape.rank()) + ") ok",
            "",
            "swap shape[" + to_string(dim0) + "]=" + to_string(shape[dim0]) +
                " <-> shape[" + to_string(dim1) + "]=" + to_string(shape[dim1]),
            "swap stride[" + to_string(dim0) + "]=" + to_string(stride[dim0]) +
                " <-> stride[" + to_string(dim1) + "]=" + to_string(stride[dim1]),
            "",
            "result.shape_ = [" + new_shape_str + "]",
            "result.stride_ = [" + new_stride_str + "]",
            "result.offset_ = unchanged",
            "",
            "is_contiguous() = false",
            "  strides not in descending order",
            "  -> TensorIterator will use strided path",
            "  -> SIMD kernels fall back to scalar",
            "",
            "NO allocation. NO copy. Same buffer.",
        };

        tracer.record("Tensor::transpose", "transpose(" + to_string(dim0) + ", " +
            to_string(dim1) + ") -> [" + new_shape_str + "]",
            "Non-contiguous view, strides swapped", details);

    } catch (const exception& e) {
        tracer.record("Error", "Exception", e.what(), {e.what()});
    }

    auto result = flushTracer();
    tracer.disable();
    return result;
}

// Trace broadcast
vector<JSTraceStep> traceBroadcast(vector<int64_t> dims_a, vector<int64_t> dims_b) {
    auto& tracer = Tracer::instance();
    tracer.enable();
    tracer.clear();

    try {
        Shape a(span<const int64_t>(dims_a.data(), dims_a.size()));
        Shape b(span<const int64_t>(dims_b.data(), dims_b.size()));

        // BroadcastEngine::compute has its own tracer calls
        auto result = BroadcastEngine::compute(a, b);

        if (holds_alternative<string>(result)) {
            auto& err = get<string>(result);
            tracer.record("BroadcastEngine", "FAILED", err, {err});
        }

    } catch (const exception& e) {
        tracer.record("Error", "Exception", e.what(), {e.what()});
    }

    auto out = flushTracer();
    tracer.disable();
    return out;
}

// Trace iterator behavior
vector<JSTraceStep> traceIterator(vector<int64_t> dims, bool transposed) {
    auto& tracer = Tracer::instance();
    tracer.enable();
    tracer.clear();

    try {
        Shape shape(span<const int64_t>(dims.data(), dims.size()));
        Tensor<DType::Float32> t(shape);

        // Fill with sequential values
        for (int64_t i = 0; i < t.numel(); ++i) {
            t.flat(i) = static_cast<float>(i);
        }

        Tensor<DType::Float32> target = t;
        if (transposed && shape.rank() >= 2) {
            target = t.transpose(0, 1);
        }

        bool is_contig = target.is_contiguous();
        string path = is_contig ? "contiguous (pointer++)" : "strided (carry propagation)";

        vector<string> details;
        details.push_back("is_contiguous() = " + string(is_contig ? "true" : "false"));
        details.push_back("iteration path: " + path);
        details.push_back("numel = " + to_string(target.numel()));
        details.push_back("");

        if (is_contig) {
            details.push_back("fast path: simple pointer increment");
            details.push_back("  for each element: ++ptr (single instruction)");
            details.push_back("  no index math, no branching");
        } else {
            details.push_back("strided path: multi-dim index with carry");
            details.push_back("  for each element:");
            details.push_back("    increment last dim index");
            details.push_back("    if overflow: reset, carry to next dim");
            details.push_back("    advance ptr by stride[dim]");
        }

        details.push_back("");
        details.push_back("first 8 elements visited:");

        auto it = TensorIterator<DType::Float32>(
            target.data(), target.shape(), target.stride(), is_contig);
        auto end = TensorIterator<DType::Float32>::end_for(
            target.data(), static_cast<size_t>(target.numel()));

        int count = 0;
        while (it != end && count < 8) {
            details.push_back("  [" + to_string(count) + "] value = " + to_string(*it));
            ++it;
            ++count;
        }
        if (target.numel() > 8) {
            details.push_back("  ... (" + to_string(target.numel() - 8) + " more)");
        }

        string shape_str;
        for (size_t i = 0; i < target.shape().rank(); ++i) {
            if (i) shape_str += ",";
            shape_str += to_string(target.shape()[i]);
        }

        tracer.record("TensorIterator", "Iterate [" + shape_str + "] (" + path + ")",
            to_string(target.numel()) + " elements, " + path, details);

    } catch (const exception& e) {
        tracer.record("Error", "Exception", e.what(), {e.what()});
    }

    auto result = flushTracer();
    tracer.disable();
    return result;
}

// Trace contiguous() on non-contiguous tensor
vector<JSTraceStep> traceContiguous(vector<int64_t> dims) {
    auto& tracer = Tracer::instance();
    tracer.enable();
    tracer.clear();

    try {
        Shape shape(span<const int64_t>(dims.data(), dims.size()));

        string dim_str;
        for (size_t i = 0; i < shape.rank(); ++i) { if (i) dim_str += ","; dim_str += to_string(shape[i]); }

        // Simulate transpose then contiguous
        Stride original_stride = Stride::from_shape(shape);
        vector<int64_t> t_dims(shape.rank()), t_strides(shape.rank());
        for (size_t i = 0; i < shape.rank(); ++i) { t_dims[i] = shape[i]; t_strides[i] = original_stride[i]; }
        if (shape.rank() >= 2) {
            swap(t_dims[0], t_dims[1]);
            swap(t_strides[0], t_strides[1]);
        }

        string t_shape_str, t_stride_str;
        for (size_t i = 0; i < shape.rank(); ++i) {
            if (i) { t_shape_str += ","; t_stride_str += ","; }
            t_shape_str += to_string(t_dims[i]);
            t_stride_str += to_string(t_strides[i]);
        }

        Shape new_shape(span<const int64_t>(t_dims.data(), shape.rank()));
        Stride new_row_stride = Stride::from_shape(new_shape);
        string new_stride_str;
        for (size_t i = 0; i < new_shape.rank(); ++i) {
            if (i) new_stride_str += ",";
            new_stride_str += to_string(new_row_stride[i]);
        }

        size_t bytes = static_cast<size_t>(shape.numel()) * 4;

        vector<string> details = {
            "input: transposed tensor, shape=[" + t_shape_str + "], stride=[" + t_stride_str + "]",
            "is_contiguous() = false (strides != row-major order)",
            "",
            "must allocate new storage and copy in row-major order:",
            "  1. allocate TensorStorage(" + to_string(bytes) + " bytes, align=16)",
            "  2. iterate source in logical order (strided iterator)",
            "  3. write sequentially to new buffer (row-major)",
            "",
            "result.storage_ = NEW shared_ptr (independent)",
            "result.shape_ = [" + t_shape_str + "]",
            "result.stride_ = [" + new_stride_str + "] (fresh row-major)",
            "result.offset_ = 0",
            "result.is_contiguous() = true",
            "",
            "This IS a data copy. New allocation required.",
        };

        tracer.record("Tensor::contiguous", "contiguous() -> new storage",
            "Copy " + to_string(shape.numel()) + " elements to contiguous layout", details);

    } catch (const exception& e) {
        tracer.record("Error", "Exception", e.what(), {e.what()});
    }

    auto result = flushTracer();
    tracer.disable();
    return result;
}

// Trace clone
vector<JSTraceStep> traceClone(vector<int64_t> dims) {
    auto& tracer = Tracer::instance();
    tracer.enable();
    tracer.clear();

    try {
        Shape shape(span<const int64_t>(dims.data(), dims.size()));
        size_t bytes = static_cast<size_t>(shape.numel()) * 4;
        size_t aligned = ((bytes + 15) / 16) * 16;

        string dim_str;
        for (size_t i = 0; i < shape.rank(); ++i) { if (i) dim_str += ","; dim_str += to_string(shape[i]); }

        Stride stride = Stride::from_shape(shape);
        string stride_str;
        for (size_t i = 0; i < stride.rank(); ++i) { if (i) stride_str += ","; stride_str += to_string(stride[i]); }

        vector<string> details = {
            "allocate new TensorStorage(" + to_string(bytes) + ", align=16)",
            "  aligned_alloc(16, " + to_string(aligned) + ") -> new pointer",
            "  memcpy(new_ptr, old_ptr, " + to_string(bytes) + ")",
            "",
            "result.storage_ = NEW shared_ptr (refcount=1, independent)",
            "result.shape_ = [" + dim_str + "]",
            "result.stride_ = [" + stride_str + "]",
            "result.offset_ = 0",
            "",
            "result.data() != original.data()",
            "modifying clone does NOT affect original",
            "",
            "This IS a deep copy. New allocation.",
        };

        tracer.record("Tensor::clone", "clone() -> independent copy",
            "Deep copy " + to_string(shape.numel()) + " elements (" + to_string(bytes) + " bytes)", details);

    } catch (const exception& e) {
        tracer.record("Error", "Exception", e.what(), {e.what()});
    }

    auto result = flushTracer();
    tracer.disable();
    return result;
}

// Embind registrations
EMSCRIPTEN_BINDINGS(inferx_wasm) {
    register_vector<string>("VectorString");
    register_vector<int64_t>("VectorInt64");

    value_object<JSTraceStep>("TraceStep")
        .field("component", &JSTraceStep::component)
        .field("title", &JSTraceStep::title)
        .field("detail", &JSTraceStep::detail)
        .field("internal", &JSTraceStep::internal);

    register_vector<JSTraceStep>("VectorTraceStep");

    emscripten::function("traceFullCreate", &traceFullCreate);
    emscripten::function("traceAccess", &traceAccess);
    emscripten::function("traceReshape", &traceReshape);
    emscripten::function("traceSlice", &traceSlice);
    emscripten::function("traceTranspose", &traceTranspose);
    emscripten::function("traceBroadcast", &traceBroadcast);
    emscripten::function("traceIterator", &traceIterator);
    emscripten::function("traceContiguous", &traceContiguous);
    emscripten::function("traceClone", &traceClone);
}
