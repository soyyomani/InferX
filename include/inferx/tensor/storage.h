#pragma once

#include <cstddef>
#include <cstdint>

namespace inferx::tensor {

class TensorStorage {
public:
    static constexpr size_t kDefaultAlignment = 16;

    explicit TensorStorage(size_t size_bytes, size_t alignment = kDefaultAlignment);
    ~TensorStorage();

    // Non-copyable
    TensorStorage(const TensorStorage&) = delete;
    TensorStorage& operator=(const TensorStorage&) = delete;

    // Moveable
    TensorStorage(TensorStorage&& other) noexcept;
    TensorStorage& operator=(TensorStorage&& other) noexcept;

    // Raw access
    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }
    [[nodiscard]] size_t size_bytes() const noexcept { return size_bytes_; }
    [[nodiscard]] size_t alignment() const noexcept { return alignment_; }

    // Typed access
    template <typename T>
    [[nodiscard]] T* data_as() noexcept { return static_cast<T*>(data_); }

    template <typename T>
    [[nodiscard]] const T* data_as() const noexcept { return static_cast<const T*>(data_); }

private:
    void* data_ = nullptr;
    size_t size_bytes_ = 0;
    size_t alignment_ = kDefaultAlignment;
};

} // namespace inferx::tensor
