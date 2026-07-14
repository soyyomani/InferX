#include <gtest/gtest.h>
#include <inferx/tensor/tensor.h>
#include <vector>

using namespace std;
using namespace inferx::tensor;

using F32Tensor = Tensor<DType::Float32>;

// Construction

TEST(TensorTest, ConstructWithShape) {
    F32Tensor t(Shape{2, 3, 4});
    EXPECT_EQ(t.rank(), 3);
    EXPECT_EQ(t.numel(), 24);
    EXPECT_EQ(t.size_bytes(), 96);
    EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorTest, Zeros) {
    auto t = F32Tensor::zeros(Shape{3, 3});
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(t.flat(i), 0.0f);
    }
}

TEST(TensorTest, Ones) {
    auto t = F32Tensor::ones(Shape{2, 2});
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(t.flat(i), 1.0f);
    }
}

TEST(TensorTest, Full) {
    auto t = F32Tensor::full(Shape{4}, 42.0f);
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(t.flat(i), 42.0f);
    }
}

TEST(TensorTest, ConstructFromSpan) {
    vector<float> data = {1, 2, 3, 4, 5, 6};
    F32Tensor t(Shape{2, 3}, data);
    EXPECT_FLOAT_EQ(t(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(t(1, 2), 6.0f);
}

TEST(TensorTest, ConstructFromSpanSizeMismatchThrows) {
    vector<float> data = {1, 2, 3};
    EXPECT_THROW(F32Tensor(Shape{2, 3}, data), invalid_argument);
}

// Element access

TEST(TensorTest, ElementAccessReadWrite) {
    auto t = F32Tensor::zeros(Shape{3, 4});
    t(1, 2) = 7.5f;
    EXPECT_FLOAT_EQ(t(1, 2), 7.5f);
    EXPECT_FLOAT_EQ(t(0, 0), 0.0f);
}

TEST(TensorTest, FlatAccess) {
    auto t = F32Tensor::zeros(Shape{6});
    t.flat(3) = 99.0f;
    EXPECT_FLOAT_EQ(t.flat(3), 99.0f);
}

// Copy shares storage

TEST(TensorTest, CopySharesStorage) {
    auto t = F32Tensor::ones(Shape{4, 4});
    F32Tensor copy = t;
    EXPECT_EQ(copy.data(), t.data());
    copy(0, 0) = 99.0f;
    EXPECT_FLOAT_EQ(t(0, 0), 99.0f); // shared
}

// Clone produces independent copy

TEST(TensorTest, CloneIndependent) {
    auto t = F32Tensor::ones(Shape{3, 3});
    auto c = t.clone();
    EXPECT_NE(c.data(), t.data());
    c(0, 0) = 99.0f;
    EXPECT_FLOAT_EQ(t(0, 0), 1.0f); // original unchanged
}

// Move semantics

TEST(TensorTest, MovePreservesData) {
    auto t = F32Tensor::full(Shape{4}, 5.0f);
    float* original_data = t.data();
    F32Tensor moved(move(t));
    EXPECT_EQ(moved.data(), original_data);
}

// Reshape

TEST(TensorTest, ReshapeZeroCopy) {
    auto t = F32Tensor::ones(Shape{6, 4});
    auto r = t.reshape(Shape{2, 3, 4});
    EXPECT_EQ(r.data(), t.data()); // shared storage
    EXPECT_EQ(r.numel(), 24);
    EXPECT_EQ(r.shape()[0], 2);
}

TEST(TensorTest, ReshapeMismatchThrows) {
    auto t = F32Tensor::ones(Shape{6, 4});
    EXPECT_THROW(t.reshape(Shape{5, 5}), invalid_argument);
}

// Slice

TEST(TensorTest, SliceSharesStorage) {
    auto t = F32Tensor::zeros(Shape{4, 6});
    t(2, 3) = 42.0f;
    auto s = t.slice(0, 2, 4);
    EXPECT_EQ(s.shape()[0], 2);
    EXPECT_EQ(s.shape()[1], 6);
    EXPECT_FLOAT_EQ(s(0, 3), 42.0f);
}

TEST(TensorTest, SliceNegativeIndices) {
    auto t = F32Tensor::zeros(Shape{10});
    t.flat(8) = 1.0f;
    auto s = t.slice(0, -3, -1);
    EXPECT_EQ(s.shape()[0], 2);
}

// Transpose

TEST(TensorTest, TransposeSwapsDims) {
    vector<float> data = {1, 2, 3, 4, 5, 6};
    F32Tensor t(Shape{2, 3}, data);
    auto tr = t.transpose(0, 1);
    EXPECT_EQ(tr.shape()[0], 3);
    EXPECT_EQ(tr.shape()[1], 2);
    EXPECT_FALSE(tr.is_contiguous());
    EXPECT_FLOAT_EQ(tr(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(tr(1, 0), 2.0f); // was t(0,1)
    EXPECT_FLOAT_EQ(tr(0, 1), 4.0f); // was t(1,0)
}

TEST(TensorTest, TransposeInvolution) {
    auto t = F32Tensor::ones(Shape{2, 3, 4});
    auto tt = t.transpose(0, 2).transpose(0, 2);
    EXPECT_EQ(tt.shape()[0], 2);
    EXPECT_EQ(tt.shape()[1], 3);
    EXPECT_EQ(tt.shape()[2], 4);
}

// Contiguous

TEST(TensorTest, ContiguousOnContiguousReturnsSame) {
    auto t = F32Tensor::ones(Shape{3, 4});
    auto c = t.contiguous();
    EXPECT_EQ(c.data(), t.data());
}

TEST(TensorTest, ContiguousOnNonContiguousCopies) {
    vector<float> data = {1, 2, 3, 4, 5, 6};
    F32Tensor t(Shape{2, 3}, data);
    auto tr = t.transpose(0, 1);
    EXPECT_FALSE(tr.is_contiguous());
    auto c = tr.contiguous();
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_NE(c.data(), t.data());
    // Verify data is correct in row-major order of transposed shape [3,2]
    EXPECT_FLOAT_EQ(c(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(c(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(c(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(c(1, 1), 5.0f);
}

// Alignment

TEST(TensorTest, DataAligned16) {
    auto t = F32Tensor::zeros(Shape{100});
    uintptr_t addr = reinterpret_cast<uintptr_t>(t.data());
    EXPECT_EQ(addr % 16, 0);
}
