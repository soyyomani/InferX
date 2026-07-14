#include <gtest/gtest.h>
#include <inferx/tensor/storage.h>
#include <cstdint>

using namespace std;
using namespace inferx::tensor;

// Allocation

TEST(StorageTest, AllocatesNonNull) {
    TensorStorage s(64);
    EXPECT_NE(s.data(), nullptr);
    EXPECT_GE(s.size_bytes(), 64);
}

TEST(StorageTest, DefaultAlignment16) {
    TensorStorage s(100);
    uintptr_t addr = reinterpret_cast<uintptr_t>(s.data());
    EXPECT_EQ(addr % 16, 0);
}

TEST(StorageTest, CustomAlignment32) {
    TensorStorage s(100, 32);
    uintptr_t addr = reinterpret_cast<uintptr_t>(s.data());
    EXPECT_EQ(addr % 32, 0);
}

TEST(StorageTest, ZeroInitialized) {
    TensorStorage s(128);
    auto* bytes = static_cast<uint8_t*>(s.data());
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(bytes[i], 0);
    }
}

// Validation

TEST(StorageTest, RejectsZeroSize) {
    EXPECT_THROW(TensorStorage(0), invalid_argument);
}

TEST(StorageTest, RejectsNonPowerOf2Alignment) {
    EXPECT_THROW(TensorStorage(64, 3), invalid_argument);
}

TEST(StorageTest, RejectsTooSmallAlignment) {
    EXPECT_THROW(TensorStorage(64, 2), invalid_argument); // < sizeof(void*)
}

// Move semantics

TEST(StorageTest, MoveConstructor) {
    TensorStorage a(256);
    void* original_ptr = a.data();
    TensorStorage b(move(a));
    EXPECT_EQ(b.data(), original_ptr);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size_bytes(), 0);
}

TEST(StorageTest, MoveAssignment) {
    TensorStorage a(128);
    TensorStorage b(64);
    void* a_ptr = a.data();
    b = move(a);
    EXPECT_EQ(b.data(), a_ptr);
    EXPECT_EQ(a.data(), nullptr);
}

// Typed access

TEST(StorageTest, TypedAccess) {
    TensorStorage s(16 * sizeof(float));
    float* fp = s.data_as<float>();
    EXPECT_NE(fp, nullptr);
    fp[0] = 3.14f;
    EXPECT_FLOAT_EQ(s.data_as<float>()[0], 3.14f);
}
