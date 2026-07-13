#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <concepts>

namespace inferx::tensor {

enum class DType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    Int8    = 2,
    Int32   = 3,
    Int64   = 4,
    Bool    = 5
};

// Compile-time type traits

template <DType D>
struct DTypeTraits;

template <>
struct DTypeTraits<DType::Float32> {
    using type = float;
    static constexpr size_t size = 4;
    static constexpr size_t alignment = 4;
    static constexpr const char* name = "float32";
};

template <>
struct DTypeTraits<DType::Float16> {
    using type = uint16_t;
    static constexpr size_t size = 2;
    static constexpr size_t alignment = 2;
    static constexpr const char* name = "float16";
};

template <>
struct DTypeTraits<DType::Int8> {
    using type = int8_t;
    static constexpr size_t size = 1;
    static constexpr size_t alignment = 1;
    static constexpr const char* name = "int8";
};

template <>
struct DTypeTraits<DType::Int32> {
    using type = int32_t;
    static constexpr size_t size = 4;
    static constexpr size_t alignment = 4;
    static constexpr const char* name = "int32";
};

template <>
struct DTypeTraits<DType::Int64> {
    using type = int64_t;
    static constexpr size_t size = 8;
    static constexpr size_t alignment = 8;
    static constexpr const char* name = "int64";
};

template <>
struct DTypeTraits<DType::Bool> {
    using type = uint8_t;
    static constexpr size_t size = 1;
    static constexpr size_t alignment = 1;
    static constexpr const char* name = "bool";
};

// Concept for valid tensor element types

template <typename T>
concept TensorScalar = std::is_arithmetic_v<T> || std::is_same_v<T, uint8_t>;

// Native type to DType mapping

template <typename T>
struct NativeToDType;

template <> struct NativeToDType<float>    { static constexpr DType value = DType::Float32; };
template <> struct NativeToDType<uint16_t> { static constexpr DType value = DType::Float16; };
template <> struct NativeToDType<int8_t>   { static constexpr DType value = DType::Int8; };
template <> struct NativeToDType<int32_t>  { static constexpr DType value = DType::Int32; };
template <> struct NativeToDType<int64_t>  { static constexpr DType value = DType::Int64; };
template <> struct NativeToDType<uint8_t>  { static constexpr DType value = DType::Bool; };

template <typename T>
inline constexpr DType native_to_dtype_v = NativeToDType<T>::value;

// Runtime size lookup

constexpr size_t dtype_size(DType d) noexcept {
    switch (d) {
        case DType::Float32: return DTypeTraits<DType::Float32>::size;
        case DType::Float16: return DTypeTraits<DType::Float16>::size;
        case DType::Int8:    return DTypeTraits<DType::Int8>::size;
        case DType::Int32:   return DTypeTraits<DType::Int32>::size;
        case DType::Int64:   return DTypeTraits<DType::Int64>::size;
        case DType::Bool:    return DTypeTraits<DType::Bool>::size;
    }
    return 0;
}

constexpr const char* dtype_name(DType d) noexcept {
    switch (d) {
        case DType::Float32: return DTypeTraits<DType::Float32>::name;
        case DType::Float16: return DTypeTraits<DType::Float16>::name;
        case DType::Int8:    return DTypeTraits<DType::Int8>::name;
        case DType::Int32:   return DTypeTraits<DType::Int32>::name;
        case DType::Int64:   return DTypeTraits<DType::Int64>::name;
        case DType::Bool:    return DTypeTraits<DType::Bool>::name;
    }
    return "unknown";
}

} // namespace inferx::tensor
