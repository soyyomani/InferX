#pragma once

#include <cstdint>
#include <cstddef>

namespace inferx::tensor {

enum class DType : uint8_t;

template <DType D>
struct DTypeTraits;

class Shape;
class Stride;
class TensorStorage;

template <DType D>
class Tensor;

template <DType D>
class TensorView;

template <DType D>
class TensorIterator;

class BroadcastEngine;

} // namespace inferx::tensor
