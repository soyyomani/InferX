# InferX

A high-performance C++20 AI inference engine built from scratch, targeting Apple Silicon (ARM64) with a focus on memory efficiency, cache-friendly layouts, and SIMD-ready architecture.

Includes an interactive **WebAssembly visualizer** that runs the actual compiled C++ tensor engine in the browser — letting you see step-by-step how tensors are created, stored, accessed, and transformed.

## Tensor Engine

The complete tensor engine is implemented with 113 unit tests passing.

**DType System** (`dtype.h`)
- Compile-time type traits mapping DType enum to native C++ types, sizes, alignment
- `TensorScalar` concept constraining valid element types
- Bidirectional type mapping (C++ type ↔ DType enum)
- Constexpr runtime size/name lookup

**Shape** (`shape.h`, `shape.cpp`)
- Immutable, stack-allocated `std::array<int64_t, 8>` (zero heap allocations, max rank 8)
- Validation, numel computation, squeeze/unsqueeze/permute
- NumPy-style broadcast compatibility checking

**Stride** (`stride.h`, `stride.cpp`)
- Row-major and column-major stride computation
- Multi-dimensional index → flat offset calculation
- Contiguity detection

**TensorStorage** (`storage.h`, `storage.cpp`)
- 16-byte aligned allocation (ARM NEON ready)
- RAII ownership, move-only semantics
- Zero-initialization, size validation

**Tensor** (`tensor.h`)
- Template class `Tensor<DType>` with shared storage via `shared_ptr`
- Named constructors: `zeros()`, `ones()`, `full()`
- Type-safe variadic element access with debug bounds checking
- Zero-copy operations: `reshape`, `slice`, `transpose`
- `contiguous()` for making non-contiguous tensors SIMD-friendly
- `clone()` for deep independent copies

**TensorView** (`tensor_view.h`)
- Non-owning read-only view (no refcount overhead)
- Implicit conversion from Tensor
- Slicing support

**BroadcastEngine** (`broadcast.h`, `broadcast.cpp`)
- NumPy-style broadcasting with virtual strides (stride=0 for broadcast dims)
- Reports incompatible dimensions with descriptive errors

**TensorIterator** (`iterator.h`)
- Dual-path iteration: pointer increment for contiguous, carry-propagation for strided
- Satisfies `std::forward_iterator` concept

## WASM Visualizer

The entire C++ tensor engine is compiled to WebAssembly via Emscripten. A React frontend renders the step-by-step trace output — the frontend has zero computation logic; everything runs in compiled C++.

Features:
- Interactive tensor builder: define shape, dtype, fill mode, chain operations
- Step-by-step pipeline view showing DType → Shape → Stride → Storage → Tensor
- 3D tensor grid (Three.js) with animated stride walk-through
- Real-world examples: RGB image, sentence embedding, weight matrix, batch of vectors
- Preset examples: create, access, reshape, slice, transpose, broadcast, iterate, clone

```bash
# Run the visualizer
cd visualizer && npm install && npm run dev
# Opens at http://localhost:5173/
```

## Build

```bash
# C++ build + tests
cmake -B build -S . -DINFERX_BUILD_TESTS=ON
cmake --build build

# Run all tests (113 total)
./build/tests/tensor/test_dtype
./build/tests/tensor/test_shape
./build/tests/tensor/test_stride
./build/tests/tensor/test_storage
./build/tests/tensor/test_tensor
./build/tests/tensor/test_broadcast
./build/tests/tensor/test_iterator

# WASM build (requires Emscripten)
source ~/emsdk/emsdk_env.sh
cd wasm && emcmake cmake -B build -S . && cmake --build build
cp build/inferx_wasm.{js,wasm} ../visualizer/public/
```

## Tests

| Component | Tests |
|-----------|-------|
| DType | 10 |
| Shape | 45 |
| Stride | 14 |
| TensorStorage | 10 |
| Tensor | 20 |
| BroadcastEngine | 9 |
| TensorIterator | 5 |
| **Total** | **113** |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | C++20 |
| Build | CMake 3.20+ |
| Compiler | Apple Clang 15 |
| Testing | GoogleTest 1.14 |
| SIMD | ARM NEON (16-byte aligned storage) |
| WASM | Emscripten 6.0 + embind |
| Visualizer | React 19 + Three.js + Vite |
| Platform | macOS ARM64 |

## Project Structure

```
InferX/
├── CMakeLists.txt
├── include/inferx/tensor/
│   ├── tensor_fwd.h       # Forward declarations
│   ├── dtype.h            # Type system
│   ├── shape.h            # Shape class
│   ├── stride.h           # Stride class
│   ├── storage.h          # Aligned memory storage
│   ├── tensor.h           # Tensor template class
│   ├── tensor_view.h      # Non-owning view
│   ├── broadcast.h        # BroadcastEngine
│   ├── iterator.h         # TensorIterator
│   └── tracer.h           # Instrumentation for visualizer
├── src/tensor/
│   ├── shape.cpp
│   ├── stride.cpp
│   ├── storage.cpp
│   └── broadcast.cpp
├── tests/tensor/
│   ├── test_dtype.cpp
│   ├── test_shape.cpp
│   ├── test_stride.cpp
│   ├── test_storage.cpp
│   ├── test_tensor.cpp
│   ├── test_broadcast.cpp
│   └── test_iterator.cpp
├── wasm/
│   ├── CMakeLists.txt     # Emscripten build
│   └── bindings.cpp       # embind bindings
├── visualizer/
│   ├── src/
│   │   ├── components/    # React UI components
│   │   └── engine/wasm.js # WASM loader
│   └── public/
│       ├── inferx_wasm.js
│       └── inferx_wasm.wasm
└── benchmarks/tensor/
    └── CMakeLists.txt
```

## Roadmap

Next: Computational graph, operator kernels (Add, MatMul, ReLU), ARM NEON SIMD backend, thread pool scheduler, ONNX model loading.

## License

MIT
