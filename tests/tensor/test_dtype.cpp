#include <gtest/gtest.h>
#include <inferx/tensor/dtype.h>
#include <string>

using namespace std;
using namespace inferx::tensor;

// Size and alignment

TEST(DTypeTest, TraitsSizeValues) {
    EXPECT_EQ(DTypeTraits<DType::Float32>::size, 4);
    EXPECT_EQ(DTypeTraits<DType::Float16>::size, 2);
    EXPECT_EQ(DTypeTraits<DType::Int8>::size, 1);
    EXPECT_EQ(DTypeTraits<DType::Int32>::size, 4);
    EXPECT_EQ(DTypeTraits<DType::Int64>::size, 8);
    EXPECT_EQ(DTypeTraits<DType::Bool>::size, 1);
}

TEST(DTypeTest, TraitsAlignmentValues) {
    EXPECT_EQ(DTypeTraits<DType::Float32>::alignment, 4);
    EXPECT_EQ(DTypeTraits<DType::Float16>::alignment, 2);
    EXPECT_EQ(DTypeTraits<DType::Int8>::alignment, 1);
    EXPECT_EQ(DTypeTraits<DType::Int32>::alignment, 4);
    EXPECT_EQ(DTypeTraits<DType::Int64>::alignment, 8);
    EXPECT_EQ(DTypeTraits<DType::Bool>::alignment, 1);
}

TEST(DTypeTest, TraitsNameStrings) {
    EXPECT_STREQ(DTypeTraits<DType::Float32>::name, "float32");
    EXPECT_STREQ(DTypeTraits<DType::Float16>::name, "float16");
    EXPECT_STREQ(DTypeTraits<DType::Int8>::name, "int8");
    EXPECT_STREQ(DTypeTraits<DType::Int32>::name, "int32");
    EXPECT_STREQ(DTypeTraits<DType::Int64>::name, "int64");
    EXPECT_STREQ(DTypeTraits<DType::Bool>::name, "bool");
}

// Native type mapping

TEST(DTypeTest, TraitsNativeTypes) {
    static_assert(is_same_v<DTypeTraits<DType::Float32>::type, float>);
    static_assert(is_same_v<DTypeTraits<DType::Float16>::type, uint16_t>);
    static_assert(is_same_v<DTypeTraits<DType::Int8>::type, int8_t>);
    static_assert(is_same_v<DTypeTraits<DType::Int32>::type, int32_t>);
    static_assert(is_same_v<DTypeTraits<DType::Int64>::type, int64_t>);
    static_assert(is_same_v<DTypeTraits<DType::Bool>::type, uint8_t>);
}

// TensorScalar concept

TEST(DTypeTest, TensorScalarAcceptsValidTypes) {
    static_assert(TensorScalar<float>);
    static_assert(TensorScalar<double>);
    static_assert(TensorScalar<int8_t>);
    static_assert(TensorScalar<int16_t>);
    static_assert(TensorScalar<int32_t>);
    static_assert(TensorScalar<int64_t>);
    static_assert(TensorScalar<uint8_t>);
    static_assert(TensorScalar<uint16_t>);
    static_assert(TensorScalar<uint32_t>);
    static_assert(TensorScalar<uint64_t>);
    static_assert(TensorScalar<bool>);
    static_assert(TensorScalar<char>);
}

TEST(DTypeTest, TensorScalarRejectsInvalidTypes) {
    static_assert(!TensorScalar<string>);
    static_assert(!TensorScalar<void*>);
    static_assert(!TensorScalar<int*>);
}

// Native to DType mapping

TEST(DTypeTest, NativeToDTypeMappingCompileTime) {
    static_assert(native_to_dtype_v<float> == DType::Float32);
    static_assert(native_to_dtype_v<uint16_t> == DType::Float16);
    static_assert(native_to_dtype_v<int8_t> == DType::Int8);
    static_assert(native_to_dtype_v<int32_t> == DType::Int32);
    static_assert(native_to_dtype_v<int64_t> == DType::Int64);
    static_assert(native_to_dtype_v<uint8_t> == DType::Bool);
}

// Runtime dtype_size

TEST(DTypeTest, RuntimeDTypeSizeMatchesTraits) {
    EXPECT_EQ(dtype_size(DType::Float32), DTypeTraits<DType::Float32>::size);
    EXPECT_EQ(dtype_size(DType::Float16), DTypeTraits<DType::Float16>::size);
    EXPECT_EQ(dtype_size(DType::Int8), DTypeTraits<DType::Int8>::size);
    EXPECT_EQ(dtype_size(DType::Int32), DTypeTraits<DType::Int32>::size);
    EXPECT_EQ(dtype_size(DType::Int64), DTypeTraits<DType::Int64>::size);
    EXPECT_EQ(dtype_size(DType::Bool), DTypeTraits<DType::Bool>::size);
}

TEST(DTypeTest, RuntimeDTypeSizeIsConstexpr) {
    constexpr size_t f32_size = dtype_size(DType::Float32);
    constexpr size_t i64_size = dtype_size(DType::Int64);
    EXPECT_EQ(f32_size, 4);
    EXPECT_EQ(i64_size, 8);
}

// Runtime dtype_name

TEST(DTypeTest, RuntimeDTypeNameMatchesTraits) {
    EXPECT_STREQ(dtype_name(DType::Float32), "float32");
    EXPECT_STREQ(dtype_name(DType::Float16), "float16");
    EXPECT_STREQ(dtype_name(DType::Int8), "int8");
    EXPECT_STREQ(dtype_name(DType::Int32), "int32");
    EXPECT_STREQ(dtype_name(DType::Int64), "int64");
    EXPECT_STREQ(dtype_name(DType::Bool), "bool");
}
