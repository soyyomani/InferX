#include <gtest/gtest.h>
#include <inferx/tensor/stride.h>
#include <vector>

using namespace std;
using namespace inferx::tensor;

// Row-major computation

TEST(StrideTest, RowMajor3D) {
    Shape s{2, 3, 4};
    Stride st = Stride::from_shape(s);
    EXPECT_EQ(st.rank(), 3);
    EXPECT_EQ(st[0], 12); // 3*4
    EXPECT_EQ(st[1], 4);  // 4
    EXPECT_EQ(st[2], 1);
}

TEST(StrideTest, RowMajor1D) {
    Shape s{10};
    Stride st = Stride::from_shape(s);
    EXPECT_EQ(st.rank(), 1);
    EXPECT_EQ(st[0], 1);
}

TEST(StrideTest, RowMajor2D) {
    Shape s{5, 8};
    Stride st = Stride::from_shape(s);
    EXPECT_EQ(st[0], 8);
    EXPECT_EQ(st[1], 1);
}

TEST(StrideTest, RowMajorScalar) {
    Shape s;
    Stride st = Stride::from_shape(s);
    EXPECT_EQ(st.rank(), 0);
}

// Column-major computation

TEST(StrideTest, ColMajor3D) {
    Shape s{2, 3, 4};
    Stride st = Stride::from_shape_col_major(s);
    EXPECT_EQ(st[0], 1);
    EXPECT_EQ(st[1], 2);  // 2
    EXPECT_EQ(st[2], 6);  // 2*3
}

TEST(StrideTest, ColMajor2D) {
    Shape s{4, 5};
    Stride st = Stride::from_shape_col_major(s);
    EXPECT_EQ(st[0], 1);
    EXPECT_EQ(st[1], 4);
}

// Offset computation

TEST(StrideTest, OffsetSimple) {
    Shape s{2, 3, 4};
    Stride st = Stride::from_shape(s);
    vector<int64_t> idx = {1, 2, 3};
    // 1*12 + 2*4 + 3*1 = 23
    EXPECT_EQ(st.offset(idx), 23);
}

TEST(StrideTest, OffsetOrigin) {
    Shape s{2, 3, 4};
    Stride st = Stride::from_shape(s);
    vector<int64_t> idx = {0, 0, 0};
    EXPECT_EQ(st.offset(idx), 0);
}

TEST(StrideTest, OffsetLastElement) {
    Shape s{2, 3, 4};
    Stride st = Stride::from_shape(s);
    vector<int64_t> idx = {1, 2, 3};
    EXPECT_EQ(st.offset(idx), 23); // last element of 24-element tensor
}

// Contiguity

TEST(StrideTest, IsContiguousRowMajor) {
    Shape s{2, 3, 4};
    Stride st = Stride::from_shape(s);
    EXPECT_TRUE(st.is_contiguous(s));
}

TEST(StrideTest, ColMajorNotContiguous) {
    Shape s{2, 3, 4};
    Stride st = Stride::from_shape_col_major(s);
    EXPECT_FALSE(st.is_contiguous(s));
}

TEST(StrideTest, SwappedStridesNotContiguous) {
    Shape s{3, 4};
    // Transpose strides: [1, 3] instead of [4, 1]
    vector<int64_t> vals = {1, 3};
    Stride st(vals);
    EXPECT_FALSE(st.is_contiguous(s));
}

// Equality

TEST(StrideTest, EqualityTrue) {
    Shape s{2, 3};
    Stride a = Stride::from_shape(s);
    Stride b = Stride::from_shape(s);
    EXPECT_EQ(a, b);
}

TEST(StrideTest, EqualityFalse) {
    Shape s1{2, 3};
    Shape s2{3, 2};
    Stride a = Stride::from_shape(s1);
    Stride b = Stride::from_shape(s2);
    EXPECT_NE(a, b);
}
