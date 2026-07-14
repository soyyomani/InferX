#include <inferx/tensor/storage.h>
#include <inferx/tensor/tracer.h>

#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

using namespace std;

namespace inferx::tensor {

TensorStorage::TensorStorage(size_t size_bytes, size_t alignment)
    : size_bytes_(size_bytes), alignment_(alignment) {

    if (size_bytes == 0) {
        throw invalid_argument("TensorStorage: size must be greater than 0");
    }
    if (alignment < sizeof(void*) || (alignment & (alignment - 1)) != 0) {
        throw invalid_argument(
            "TensorStorage: alignment must be a power of 2 and >= " +
            to_string(sizeof(void*)) + ", got " + to_string(alignment));
    }

    // Round up to alignment boundary
    size_t aligned_size = (size_bytes + alignment - 1) & ~(alignment - 1);

    data_ = aligned_alloc(alignment, aligned_size);
    if (!data_) {
        throw bad_alloc();
    }

    memset(data_, 0, aligned_size);
    size_bytes_ = aligned_size;

    auto& tracer = Tracer::instance();
    if (tracer.is_enabled()) {
        tracer.record("TensorStorage",
            "Allocate " + to_string(size_bytes) + " bytes (align=" + to_string(alignment) + "B)",
            "aligned_alloc(" + to_string(alignment) + ", " + to_string(aligned_size) + ") -> zeroed",
            {
                "requested: " + to_string(size_bytes) + " bytes",
                "alignment: " + to_string(alignment) + " bytes (ARM NEON)",
                "rounded up: " + to_string(aligned_size) + " bytes",
                "memset(ptr, 0, " + to_string(aligned_size) + ") -> zero-initialized",
                "ownership: RAII, move-only",
            });
    }
}

TensorStorage::~TensorStorage() {
    if (data_) {
        free(data_);
    }
}

TensorStorage::TensorStorage(TensorStorage&& other) noexcept
    : data_(other.data_), size_bytes_(other.size_bytes_), alignment_(other.alignment_) {
    other.data_ = nullptr;
    other.size_bytes_ = 0;
}

TensorStorage& TensorStorage::operator=(TensorStorage&& other) noexcept {
    if (this != &other) {
        if (data_) free(data_);
        data_ = other.data_;
        size_bytes_ = other.size_bytes_;
        alignment_ = other.alignment_;
        other.data_ = nullptr;
        other.size_bytes_ = 0;
    }
    return *this;
}

} // namespace inferx::tensor
