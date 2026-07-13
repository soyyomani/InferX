#include <gtest/gtest.h>
#include <inferx/tensor/shape.h>
#include <vector>

using namespace std;
using namespace inferx::tensor;

// Construction

TEST(ShapeTest, DefaultConstructCreatesScalar) {
    Shape s;
    EXPECT_EQ(s.rank(), 0);
    EXPECT_TRUE(s.is_scalar());
    EXPECT_EQ(s.numel(), 1);
}

TEST(ShapeTest, ConstructFromInitializerList) {
    Shape s{2, 3, 4};
    EXPECT_EQ(s.rank(), 3);
    EXPECT_EQ(s[0], 2);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(s[2], 4);
    EXPECT_EQ(s.numel(), 24);
}

TEST(ShapeTest, ConstructFromSpan) {
    vector<int64_t> dims = {5, 10, 20};
    auto sp = span<const int64_t>(dims);
    Shape s(sp);
    EXPECT_EQ(s.rank(), 3);
    EXPECT_EQ(s[0], 5);
    EXPECT_EQ(s[1], 10);
    EXPECT_EQ(s[2], 20);
    EXPECT_EQ(s.numel(), 1000);
}

TEST(ShapeTest, ConstructRank1) {
    Shape s{128};
    EXPECT_EQ(s.rank(), 1);
    EXPECT_EQ(s[0], 128);
    EXPECT_EQ(s.numel(), 128);
    EXPECT_FALSE(s.is_scalar());
}

TEST(ShapeTest, ConstructMaxRank) {
    Shape s{1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(s.rank(), 8);
    EXPECT_EQ(s.numel(), 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8);
}

// Validation

TEST(ShapeTest, RejectsZeroDimension) {
    EXPECT_THROW(Shape({3, 0, 4}), invalid_argument);
}

TEST(ShapeTest, RejectsNegativeDimension) {
    EXPECT_THROW(Shape({2, -1, 3}), invalid_argument);
}

TEST(ShapeTest, RejectsExceedingMaxRank) {
    EXPECT_THROW(Shape({1, 2, 3, 4, 5, 6, 7, 8, 9}), invalid_argument);
}

// Accessors

TEST(ShapeTest, AtBoundsChecked) {
    Shape s{4, 5, 6};
    EXPECT_EQ(s.at(0), 4);
    EXPECT_EQ(s.at(1), 5);
    EXPECT_EQ(s.at(2), 6);
    EXPECT_THROW(s.at(3), out_of_range);
}

TEST(ShapeTest, NumelSingleDim) {
    Shape s{42};
    EXPECT_EQ(s.numel(), 42);
}

TEST(ShapeTest, NumelLarge) {
    Shape s{64, 128, 256};
    EXPECT_EQ(s.numel(), 64 * 128 * 256);
}

// Squeeze

TEST(ShapeTest, SqueezeRemovesSizeOneDim) {
    Shape s{3, 1, 4};
    Shape squeezed = s.squeeze(1);
    EXPECT_EQ(squeezed.rank(), 2);
    EXPECT_EQ(squeezed[0], 3);
    EXPECT_EQ(squeezed[1], 4);
}

TEST(ShapeTest, SqueezeFirstDim) {
    Shape s{1, 5, 6};
    Shape squeezed = s.squeeze(0);
    EXPECT_EQ(squeezed.rank(), 2);
    EXPECT_EQ(squeezed[0], 5);
    EXPECT_EQ(squeezed[1], 6);
}

TEST(ShapeTest, SqueezeLastDim) {
    Shape s{2, 3, 1};
    Shape squeezed = s.squeeze(2);
    EXPECT_EQ(squeezed.rank(), 2);
    EXPECT_EQ(squeezed[0], 2);
    EXPECT_EQ(squeezed[1], 3);
}

TEST(ShapeTest, SqueezeRejectsNonUnitDim) {
    Shape s{3, 4, 5};
    EXPECT_THROW(s.squeeze(1), invalid_argument);
}

TEST(ShapeTest, SqueezeRejectsOutOfRange) {
    Shape s{1, 2, 3};
    EXPECT_THROW(s.squeeze(3), invalid_argument);
}

TEST(ShapeTest, SqueezeDoesNotMutateOriginal) {
    Shape s{3, 1, 4};
    [[maybe_unused]] Shape squeezed = s.squeeze(1);
    EXPECT_EQ(s.rank(), 3);
    EXPECT_EQ(s[1], 1);
}

// Unsqueeze

TEST(ShapeTest, UnsqueezeInsertsAtMiddle) {
    Shape s{3, 4};
    Shape unsqueezed = s.unsqueeze(1);
    EXPECT_EQ(unsqueezed.rank(), 3);
    EXPECT_EQ(unsqueezed[0], 3);
    EXPECT_EQ(unsqueezed[1], 1);
    EXPECT_EQ(unsqueezed[2], 4);
}

TEST(ShapeTest, UnsqueezeAtBeginning) {
    Shape s{5, 6};
    Shape unsqueezed = s.unsqueeze(0);
    EXPECT_EQ(unsqueezed.rank(), 3);
    EXPECT_EQ(unsqueezed[0], 1);
    EXPECT_EQ(unsqueezed[1], 5);
    EXPECT_EQ(unsqueezed[2], 6);
}

TEST(ShapeTest, UnsqueezeAtEnd) {
    Shape s{2, 3};
    Shape unsqueezed = s.unsqueeze(2);
    EXPECT_EQ(unsqueezed.rank(), 3);
    EXPECT_EQ(unsqueezed[0], 2);
    EXPECT_EQ(unsqueezed[1], 3);
    EXPECT_EQ(unsqueezed[2], 1);
}

TEST(ShapeTest, UnsqueezeRejectsOutOfRange) {
    Shape s{2, 3};
    EXPECT_THROW(s.unsqueeze(3), invalid_argument);
}

TEST(ShapeTest, UnsqueezeDoesNotMutateOriginal) {
    Shape s{3, 4};
    [[maybe_unused]] Shape unsqueezed = s.unsqueeze(1);
    EXPECT_EQ(s.rank(), 2);
    EXPECT_EQ(s[0], 3);
    EXPECT_EQ(s[1], 4);
}

// Permute

TEST(ShapeTest, PermuteReverseDims) {
    Shape s{2, 3, 4};
    vector<size_t> order = {2, 1, 0};
    Shape permuted = s.permute(order);
    EXPECT_EQ(permuted.rank(), 3);
    EXPECT_EQ(permuted[0], 4);
    EXPECT_EQ(permuted[1], 3);
    EXPECT_EQ(permuted[2], 2);
}

TEST(ShapeTest, PermuteIdentity) {
    Shape s{5, 6, 7};
    vector<size_t> order = {0, 1, 2};
    Shape permuted = s.permute(order);
    EXPECT_EQ(permuted, s);
}

TEST(ShapeTest, PermuteSwapTwo) {
    Shape s{10, 20, 30};
    vector<size_t> order = {0, 2, 1};
    Shape permuted = s.permute(order);
    EXPECT_EQ(permuted[0], 10);
    EXPECT_EQ(permuted[1], 30);
    EXPECT_EQ(permuted[2], 20);
}

TEST(ShapeTest, PermuteRejectsWrongSize) {
    Shape s{2, 3, 4};
    vector<size_t> order = {0, 1};
    EXPECT_THROW(s.permute(order), invalid_argument);
}

TEST(ShapeTest, PermuteRejectsDuplicateIndex) {
    Shape s{2, 3, 4};
    vector<size_t> order = {0, 0, 1};
    EXPECT_THROW(s.permute(order), invalid_argument);
}

TEST(ShapeTest, PermuteRejectsOutOfRange) {
    Shape s{2, 3, 4};
    vector<size_t> order = {0, 1, 5};
    EXPECT_THROW(s.permute(order), invalid_argument);
}

TEST(ShapeTest, PermuteDoesNotMutateOriginal) {
    Shape s{2, 3, 4};
    vector<size_t> order = {2, 0, 1};
    [[maybe_unused]] Shape permuted = s.permute(order);
    EXPECT_EQ(s[0], 2);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(s[2], 4);
}

// Broadcasting

TEST(ShapeTest, BroadcastSameShape) {
    Shape a{3, 4};
    Shape b{3, 4};
    auto result = Shape::broadcast(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, Shape({3, 4}));
}

TEST(ShapeTest, BroadcastScalarWithTensor) {
    Shape scalar;
    Shape tensor{2, 3, 4};
    auto result = Shape::broadcast(scalar, tensor);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, tensor);
}

TEST(ShapeTest, BroadcastTensorWithScalar) {
    Shape tensor{5, 6};
    Shape scalar;
    auto result = Shape::broadcast(tensor, scalar);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, tensor);
}

TEST(ShapeTest, BroadcastExpandDim1) {
    Shape a{1, 3};
    Shape b{4, 1};
    auto result = Shape::broadcast(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, Shape({4, 3}));
}

TEST(ShapeTest, BroadcastDifferentRanks) {
    Shape a{3};
    Shape b{2, 3};
    auto result = Shape::broadcast(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, Shape({2, 3}));
}

TEST(ShapeTest, BroadcastComplex) {
    Shape a{1, 3, 1};
    Shape b{4, 1, 5};
    auto result = Shape::broadcast(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, Shape({4, 3, 5}));
}

TEST(ShapeTest, BroadcastIncompatible) {
    Shape a{3, 4};
    Shape b{3, 5};
    auto result = Shape::broadcast(a, b);
    EXPECT_FALSE(result.has_value());
}

TEST(ShapeTest, BroadcastIncompatibleNeitherIs1) {
    Shape a{2, 3};
    Shape b{4, 3};
    auto result = Shape::broadcast(a, b);
    EXPECT_FALSE(result.has_value());
}

TEST(ShapeTest, BroadcastSymmetry) {
    Shape a{1, 5};
    Shape b{3, 1};
    auto ab = Shape::broadcast(a, b);
    auto ba = Shape::broadcast(b, a);
    ASSERT_TRUE(ab.has_value());
    ASSERT_TRUE(ba.has_value());
    EXPECT_EQ(*ab, *ba);
}

TEST(ShapeTest, IsBroadcastableWith) {
    Shape a{1, 3};
    Shape b{4, 1};
    EXPECT_TRUE(a.is_broadcastable_with(b));

    Shape c{2, 3};
    Shape d{4, 3};
    EXPECT_FALSE(c.is_broadcastable_with(d));
}

// Comparison

TEST(ShapeTest, EqualityTrue) {
    Shape a{2, 3, 4};
    Shape b{2, 3, 4};
    EXPECT_EQ(a, b);
}

TEST(ShapeTest, EqualityFalseDiffRank) {
    Shape a{2, 3};
    Shape b{2, 3, 4};
    EXPECT_NE(a, b);
}

TEST(ShapeTest, EqualityFalseDiffDims) {
    Shape a{2, 3};
    Shape b{2, 5};
    EXPECT_NE(a, b);
}

TEST(ShapeTest, ScalarsAreEqual) {
    Shape a;
    Shape b;
    EXPECT_EQ(a, b);
}

// Iteration

TEST(ShapeTest, IterationVisitsAllDims) {
    Shape s{2, 3, 4, 5};
    vector<int64_t> dims(s.begin(), s.end());
    ASSERT_EQ(dims.size(), 4);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 3);
    EXPECT_EQ(dims[2], 4);
    EXPECT_EQ(dims[3], 5);
}

TEST(ShapeTest, IterationScalarEmpty) {
    Shape s;
    vector<int64_t> dims(s.begin(), s.end());
    EXPECT_TRUE(dims.empty());
}
