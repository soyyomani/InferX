# InferX

A high-performance C++20 AI inference engine built from scratch, targeting Apple Silicon (ARM64) with a focus on memory efficiency, cache-friendly layouts, and SIMD-ready architecture.

## Current Status

Building the **Tensor Engine** — the foundational module that all other components depend on.

### Implemented

**DType System** (`include/inferx/tensor/dtype.h`)
- `DType` enum with uint8_t underlying type: Float32, Float16, Int8, Int32, Int64, Bool
- `DTypeTraits<D>` compile-time traits mapping each DType to its native C++ type, byte size, alignment, and name
- `TensorScalar` concept constraining valid tensor element types
- `NativeToDType<T>` reverse mapping from C++ type to DType enum value
- `dtype_size()` / `dtype_name()` constexpr runtime lookup functions

**Shape** (`include/inferx/tensor/shape.h`, `src/tensor/shape.cpp`)
- Immutable dimension representation with fixed `std::array<int64_t, 8>` storage (zero heap allocations)
- Max rank 8, covers all practical ML tensor dimensions
- Validation: rejects zero/negative dimensions and rank > 8
- `numel()` total element count, `is_scalar()` for rank-0
- `squeeze(dim)` removes a size-1 dimension
- `unsqueeze(dim)` inserts a size-1 dimension
- `permute(order)` reorders dimensions
- `broadcast(a, b)` computes NumPy-style broadcast output shape
- All operations return new Shape instances (immutable design)

### Tests

- 10 DType tests — traits, concept validation, runtime/compile-time consistency
- 45 Shape tests — construction, validation, operations, broadcasting, comparison, iteration

All passing on Apple Clang 15 with C++20.

## Build

```bash
cmake -B build -S . -DINFERX_BUILD_TESTS=ON
cmake --build build
./build/tests/tensor/test_dtype
./build/tests/tensor/test_shape
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | C++20 |
| Build | CMake 3.20+ |
| Compiler | Apple Clang |
| Testing | GoogleTest 1.14 |
| Benchmarks | Google Benchmark 1.8 |
| SIMD | ARM NEON (planned) |
| Platform | macOS ARM64 |

## Project Structure

```
InferX/
├── CMakeLists.txt
├── include/inferx/tensor/
│   ├── tensor_fwd.h      # Forward declarations
│   ├── dtype.h            # Type system
│   └── shape.h            # Shape class
├── src/tensor/
│   ├── shape.cpp          # Shape implementation
│   ├── stride.cpp         # (placeholder)
│   ├── storage.cpp        # (placeholder)
│   └── broadcast.cpp      # (placeholder)
├── tests/tensor/
│   ├── test_dtype.cpp
│   └── test_shape.cpp
└── benchmarks/tensor/
    └── CMakeLists.txt
```

## Roadmap

Next up: Stride computation, TensorStorage (aligned memory), Tensor class, TensorView, BroadcastEngine, TensorIterator.

See `roadmap.md` for the full plan.

## License

MIT
