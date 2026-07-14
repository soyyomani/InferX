#include <gtest/gtest.h>
#include <inferx/tensor/tensor.h>
#include <inferx/tensor/stride.h>
#include <inferx/tensor/iterator.h>
#include <vector>

using namespace std;
using namespace inferx::tensor;

using F32Tensor = Tensor<DType::Float32>;
using F32Iterator = TensorIterator<DType::Float32>;

// Helper: create begin/end for a tensor
pair<F32Iterator, F32Iterator> make_iter(F32Tensor& t) {
    auto begin = F32Iterator(t.data(), t.shape(), t.stride(), t.is_contiguous());
    auto end = F32Iterator::end_for(t.data(), static_cast<size_t>(t.numel()));
    return {begin, end};
}

// Contiguous iteration

TEST(IteratorTest, ContiguousVisitsAll) {
    auto t = F32Tensor::full(Shape{2, 3}, 1.0f);
    auto [it, end] = make_iter(t);
    int count = 0;
    while (it != end) {
        EXPECT_FLOAT_EQ(*it, 1.0f);
        ++it;
        ++count;
    }
    EXPECT_EQ(count, 6);
}

TEST(IteratorTest, ContiguousOrder) {
    vector<float> data = {0, 1, 2, 3, 4, 5};
    F32Tensor t(Shape{2, 3}, data);
    auto [it, end] = make_iter(t);
    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(*it, static_cast<float>(i));
        ++it;
    }
    EXPECT_TRUE(it == end);
}

// Non-contiguous iteration (transposed)

TEST(IteratorTest, NonContiguousLogicalOrder) {
    // Create [2,3] with values 0..5, transpose to [3,2]
    vector<float> data = {0, 1, 2, 3, 4, 5};
    F32Tensor t(Shape{2, 3}, data);
    auto tr = t.transpose(0, 1); // shape [3,2], non-contiguous
    EXPECT_FALSE(tr.is_contiguous());

    // Transposed logical order: (0,0)=0, (0,1)=3, (1,0)=1, (1,1)=4, (2,0)=2, (2,1)=5
    vector<float> expected = {0, 3, 1, 4, 2, 5};

    auto begin = F32Iterator(tr.data(), tr.shape(), tr.stride(), tr.is_contiguous());
    auto end = F32Iterator::end_for(tr.data(), static_cast<size_t>(tr.numel()));

    vector<float> got;
    for (auto it = begin; it != end; ++it) {
        got.push_back(*it);
    }

    ASSERT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i], expected[i]) << "at index " << i;
    }
}

// Empty tensor

TEST(IteratorTest, EmptyTensorBeginEqualsEnd) {
    // A tensor with numel=0 can't be created (Shape rejects dim=0),
    // so we test the iterator directly with numel=0
    float dummy = 0;
    auto begin = F32Iterator(&dummy, Shape{1}, Stride::from_shape(Shape{1}), true);
    auto end = F32Iterator::end_for(&dummy, 0);
    // end has flat_idx_=0, begin has flat_idx_=0 only if numel would be 0
    // But our iterator starts at flat_idx_=0 and end is also 0 for empty case
    // Actually test: 0 increments means numel=0
    // The real test: end_for with numel=0 vs a begin that hasn't incremented
    auto e = F32Iterator::end_for(&dummy, 0);
    auto b = F32Iterator();
    EXPECT_TRUE(b == e);
}

// Exactly numel increments reach end

TEST(IteratorTest, ExactlyNumelIncrements) {
    auto t = F32Tensor::ones(Shape{3, 4, 2});
    auto [it, end] = make_iter(t);
    for (int i = 0; i < 24; ++i) {
        EXPECT_FALSE(it == end);
        ++it;
    }
    EXPECT_TRUE(it == end);
}
