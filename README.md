<p align="center">
  <img src="logo/inferx-logo-light.svg" alt="InferX Logo" width="280" />
</p>

<p align="center">
  <strong>A from-scratch C++20 AI inference engine with an interactive browser visualizer</strong>
</p>

<p align="center">
  <a href="#live-demo">Live Demo</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#visualizer">Visualizer</a> •
  <a href="#build">Build</a> •
  <a href="#roadmap">Roadmap</a>
</p>

---

## What is InferX?

InferX is a **high-performance AI inference engine** built entirely from scratch in modern C++20 — no PyTorch, no TensorFlow, no external ML libraries. It includes a complete tensor engine, neural network operations (attention, matmul, softmax, embeddings, activations), and compiles to **WebAssembly** so the real math runs directly in the browser.

The project includes an **interactive visualizer** that lets you watch every single operation step by step — from tokenization to attention scores to final predictions.

## Live Demo

```bash
cd visualizer && npm install && npm run dev
# Opens at http://localhost:5173
```

## Features

### Tensor Engine (C++20)
- **Type-safe DType system** — compile-time traits, bidirectional C++ ↔ enum mapping
- **Stack-allocated Shape** — zero heap allocations, max rank 8, NumPy-style broadcasting
- **Row-major Strides** — multi-index → flat offset, contiguity detection
- **16-byte aligned Storage** — ARM NEON ready, RAII ownership, move-only
- **Tensor template class** — shared storage, named constructors (`zeros`, `ones`, `full`)
- **Zero-copy operations** — reshape, slice, transpose (metadata-only, no data movement)
- **BroadcastEngine** — NumPy-style virtual strides with descriptive error reporting
- **TensorIterator** — dual-path (pointer increment for contiguous, carry-propagation for strided)

### Neural Network Operations (C++20)
- **MatMul** — matrix multiplication with step-by-step tracing
- **Softmax** — numerically stable (max subtraction) with traced computation
- **Activations** — ReLU, GELU with traced element-wise operations
- **Attention** — full self-attention with Q/K/V projections, scaled dot-product, multi-head support
- **Embedding** — vocabulary lookup with positional encoding
- **Tokenizer** — whitespace-based tokenization with vocabulary mapping

### Interactive Visualizer (React + WebAssembly)
- **5 pages** — Landing, Text AI Pipeline, Vision AI Pipeline, Math Lab, Tensor Playground
- **Text AI Pipeline** — 6-stage transformer visualization (tokenize → embed → position → attention → FFN → predict)
- **Vision AI Pipeline** — CNN pipeline with real MNIST inference (pixels → conv → relu → pool → FC → softmax)
- **Math Lab** — interactive operations with step-by-step trace output
- **Tensor Playground** — create tensors, chain operations, watch the C++ engine execute in real time
- **3D visualization** — Three.js tensor grid with animated stride walk-through

## Quick Start

### Run the Visualizer (no C++ build needed)

```bash
cd visualizer
npm install
npm run dev
```

The visualizer ships with pre-compiled WASM — you can explore everything immediately.

### Build the C++ Engine

```bash
cmake -B build -S . -DINFERX_BUILD_TESTS=ON
cmake --build build

# Run all 113 tests
ctest --test-dir build --output-on-failure
```

### Build for WebAssembly

```bash
source ~/emsdk/emsdk_env.sh
cd src/wasm
emcmake cmake -B build -S .
cmake --build build
cp build/inferx_wasm.{js,wasm} ../../visualizer/public/
```

## Architecture

```
InferX/
├── include/inferx/
│   ├── core/
│   │   └── tracer.h              # Step-by-step instrumentation
│   ├── tensor/
│   │   ├── dtype.h               # Type system
│   │   ├── shape.h               # Shape (stack-allocated)
│   │   ├── stride.h              # Stride computation
│   │   ├── storage.h             # Aligned memory management
│   │   ├── tensor.h              # Tensor<DType> template
│   │   ├── tensor_view.h         # Non-owning read-only view
│   │   ├── broadcast.h           # NumPy-style broadcasting
│   │   └── iterator.h            # Forward iterator
│   └── nn/
│       ├── matmul.h              # Matrix multiplication
│       ├── softmax.h             # Softmax (numerically stable)
│       ├── activations.h         # ReLU, GELU
│       ├── attention.h           # Self-attention mechanism
│       ├── embedding.h           # Token + positional embedding
│       └── tokenizer.h           # Text tokenization
├── src/
│   ├── tensor/                   # Tensor engine implementation
│   └── wasm/
│       ├── CMakeLists.txt        # Emscripten build config
│       └── nn_bindings.cpp       # embind bindings for all NN ops
├── tests/tensor/                 # 113 unit tests (GoogleTest)
├── visualizer/
│   ├── src/
│   │   ├── components/           # React pages & UI
│   │   │   ├── Landing.jsx       # Home page
│   │   │   ├── TextPipeline.jsx  # Transformer visualization
│   │   │   ├── ImagePipeline.jsx # CNN visualization
│   │   │   ├── MathExplorer.jsx  # Interactive math lab
│   │   │   ├── TensorPage.jsx    # Tensor playground
│   │   │   └── visualizations/   # Heatmaps, charts, animations
│   │   └── engine/               # WASM loader + model weights
│   └── public/                   # Static assets + compiled WASM
├── logo/                         # Brand assets
└── benchmarks/                   # Performance benchmarks
```

## Tests

| Component | Tests | Status |
|-----------|-------|--------|
| DType | 10 | ✅ |
| Shape | 45 | ✅ |
| Stride | 14 | ✅ |
| TensorStorage | 10 | ✅ |
| Tensor | 20 | ✅ |
| BroadcastEngine | 9 | ✅ |
| TensorIterator | 5 | ✅ |
| **Total** | **113** | **All passing** |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core Engine | C++20 (concepts, constexpr, ranges) |
| Build System | CMake 3.20+ |
| Testing | GoogleTest 1.14 |
| SIMD | ARM NEON (16-byte aligned) |
| WASM | Emscripten 6.0 + embind |
| Frontend | React 19 + Vite 8 |
| 3D | Three.js + @react-three/fiber |
| Target | macOS ARM64, Browser (WASM) |

## Visualizer Pages

### Text AI Pipeline
Type any sentence and watch the 6-stage transformer process it:
1. **Tokenization** — split text into tokens, assign IDs from vocabulary
2. **Embedding** — lookup 768-dim meaning vectors from trained matrix
3. **Positional Encoding** — add sin/cos position signals
4. **Self-Attention** — compute Q·K^T/√d, apply softmax, aggregate values
5. **Feed-Forward Network** — expand 4×, GELU, compress (per-token)
6. **Output Prediction** — project to vocabulary, softmax → next word

### Vision AI Pipeline
Upload an image and watch the CNN classify it (real trained MNIST model):
1. **Pixels** — image as 28×28 grid of numbers [0,1]
2. **Convolution** — slide 3×3 filters to detect edges/patterns
3. **ReLU** — zero out negatives (non-linearity)
4. **Max Pooling** — downsample 2×2, keep strongest features
5. **Fully Connected** — flatten + matrix multiply → 10 logits
6. **Softmax** — probabilities, argmax → predicted digit

### Math Lab
Interactive exploration of individual operations with customizable inputs:
- Matrix Multiply, Softmax, ReLU, GELU, Attention, Tokenizer

### Tensor Playground
Build tensors from scratch, chain operations, see C++ engine trace output in real time.

## Roadmap

- [ ] Computational graph with topological execution
- [ ] Operator kernels: Add, Sub, Div, Pow, LayerNorm
- [ ] ARM NEON SIMD matmul kernel
- [ ] Thread pool scheduler for parallel execution
- [ ] ONNX model loading (inference from exported models)
- [ ] Autograd (backward pass, gradient accumulation)
- [ ] GPU backend (Metal on macOS)

## License

MIT
