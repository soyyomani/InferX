#include <gtest/gtest.h>
#include <inferx/tensor/broadcast.h>

using namespace std;
using namespace inferx::tensor;

TEST(BroadcastTest, SameShape) {
    auto result = BroadcastEngine::compute(Shape{3, 4}, Shape{3, 4});
    ASSERT_TRUE(holds_alternative<BroadcastEngine::BroadcastResult>(result));
    auto& r = get<BroadcastEngine::BroadcastResult>(result);
    EXPECT_EQ(r.output_shape, Shape({3, 4}));
    EXPECT_FALSE(r.a_needs_broadcast);
    EXPECT_FALSE(r.b_needs_broadcast);
}

TEST(BroadcastTest, ExpandDim1) {
    auto result = BroadcastEngine::compute(Shape{1, 3}, Shape{4, 1});
    ASSERT_TRUE(holds_alternative<BroadcastEngine::BroadcastResult>(result));
    auto& r = get<BroadcastEngine::BroadcastResult>(result);
    EXPECT_EQ(r.output_shape, Shape({4, 3}));
    EXPECT_TRUE(r.a_needs_broadcast);
    EXPECT_TRUE(r.b_needs_broadcast);
}

TEST(BroadcastTest, DifferentRanks) {
    auto result = BroadcastEngine::compute(Shape{3}, Shape{2, 3});
    ASSERT_TRUE(holds_alternative<BroadcastEngine::BroadcastResult>(result));
    auto& r = get<BroadcastEngine::BroadcastResult>(result);
    EXPECT_EQ(r.output_shape, Shape({2, 3}));
}

TEST(BroadcastTest, ScalarBroadcast) {
    Shape scalar;
    auto result = BroadcastEngine::compute(scalar, Shape{4, 5});
    ASSERT_TRUE(holds_alternative<BroadcastEngine::BroadcastResult>(result));
    auto& r = get<BroadcastEngine::BroadcastResult>(result);
    EXPECT_EQ(r.output_shape, Shape({4, 5}));
}

TEST(BroadcastTest, VirtualStridesZeroForBroadcast) {
    auto result = BroadcastEngine::compute(Shape{1, 4}, Shape{3, 1});
    ASSERT_TRUE(holds_alternative<BroadcastEngine::BroadcastResult>(result));
    auto& r = get<BroadcastEngine::BroadcastResult>(result);
    // A's dim 0 is broadcast (size 1 -> 3), so stride_a[0] = 0
    EXPECT_EQ(r.stride_a[0], 0);
    // B's dim 1 is broadcast (size 1 -> 4), so stride_b[1] = 0
    EXPECT_EQ(r.stride_b[1], 0);
}

TEST(BroadcastTest, Incompatible) {
    auto result = BroadcastEngine::compute(Shape{3, 4}, Shape{3, 5});
    ASSERT_TRUE(holds_alternative<string>(result));
    auto& err = get<string>(result);
    EXPECT_TRUE(err.find("not broadcastable") != string::npos);
}

TEST(BroadcastTest, Symmetry) {
    auto ab = BroadcastEngine::compute(Shape{1, 5}, Shape{3, 1});
    auto ba = BroadcastEngine::compute(Shape{3, 1}, Shape{1, 5});
    ASSERT_TRUE(holds_alternative<BroadcastEngine::BroadcastResult>(ab));
    ASSERT_TRUE(holds_alternative<BroadcastEngine::BroadcastResult>(ba));
    EXPECT_EQ(get<BroadcastEngine::BroadcastResult>(ab).output_shape,
              get<BroadcastEngine::BroadcastResult>(ba).output_shape);
}

TEST(BroadcastTest, Compatible) {
    EXPECT_TRUE(BroadcastEngine::compatible(Shape{1, 3}, Shape{4, 1}));
    EXPECT_FALSE(BroadcastEngine::compatible(Shape{2, 3}, Shape{4, 3}));
}

TEST(BroadcastTest, OutputShapeOnly) {
    auto s = BroadcastEngine::output_shape(Shape{1, 3}, Shape{4, 1});
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(*s, Shape({4, 3}));
}
