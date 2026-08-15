<p align="center">
  <img src="logo/inferx-logo-light.svg" alt="InferX Logo" width="280" />
</p>

<p align="center">
  <strong>Interactive AI education platform powered by a from-scratch C++20 inference engine</strong>
</p>

<p align="center">
  <a href="#live-demo">Live Demo</a> •
  <a href="#what-you-learn">What You Learn</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#roadmap">Roadmap</a>
</p>

---

## What is InferX?

InferX is an **interactive AI education platform** where you watch every calculation that powers ChatGPT, Claude, and neural networks — from tokenization to attention to live inference. Built on a **high-performance C++20 inference engine** compiled to WebAssembly, with zero external ML dependencies.

Not another textbook. Not another video. You type a prompt, and you see the math happen.

## Live Demo

```bash
cd visualizer && npm install && npm run dev
```

Opens at `http://localhost:5173` — no build step needed for the C++ engine (ships pre-compiled as WASM).

## What You Learn

The platform follows a **sequential 4-module path** — each module unlocks after completing the previous:

| # | Module | What it covers | Time |
|---|--------|---------------|------|
| 1 | **Math Lab** | MatMul, Softmax, ReLU, GELU, Tokenization, Attention — 6 operations with interactive step-by-step traces and quizzes | 15 min |
| 2 | **How Text AI Works** | Complete 8-stage transformer pipeline: Tokenization → Embedding → Positional Encoding → Self-Attention → Feed-Forward → Output Prediction → Transformer Stack (LLM scaling) → RAG | 40 min |
| 3 | **How Vision AI Works** | CNN pipeline with real trained model: Pixels → Convolution → ReLU → MaxPool → FC → Softmax. Upload your own images. | 20 min |
| 4 | **MNIST Playground** | Draw digits with your mouse, watch neurons fire layer by layer, get real-time predictions from the trained model | 5 min |

Routes: `/math`, `/textai`, `/visionai`, `/mnist`

## Features

### C++20 Inference Engine
- Custom tensor engine with zero-copy operations, broadcasting, iterators
- Neural network ops: MatMul, Softmax, ReLU, GELU, Attention, Embedding, Tokenizer
- ARM NEON SIMD kernels (22 GFLOPS on Apple Silicon)
- Arena memory allocator (918x faster than malloc)
- 216+ unit tests, all passing
- Compiles to WebAssembly via Emscripten — runs in the browser

### Interactive Visualizer
- **Type your own input** — see YOUR text tokenized, embedded, attended to, and predicted
- **8-stage text pipeline** — covers everything from basic tokenization through to RAG
- **Real trained model** — Vision AI uses an actual MNIST model (98.5% accuracy)
- **C++ source shown** — every stage displays the actual engine code (tokenizer.h, attention.h, softmax.h)
- **Sequential progression** — complete each module to unlock the next
- **Persistent progress** — saved in localStorage, survives page refresh

### What's Real vs Simulated

| Component | Status | Details |
|-----------|--------|---------|
| MatMul, Softmax, ReLU, GELU | Real | C++ engine via WASM |
| BPE Tokenizer | Real | C++ vocabulary lookup |
| Attention (Q·K/√d, softmax, V) | Real | Full dot-product math |
| MNIST inference | Real | Trained model, actual weights |
| Neuron visualization | Real | From actual forward pass |
| Text embeddings | Demo | Pseudo-random (not trained) |
| Next-word predictions | Demo | Simulated (no real LLM weights) |

## Quick Start

### Run the Visualizer

```bash
cd visualizer
npm install
npm run dev
```

### Build the C++ Engine

```bash
cmake -B build -S . -DINFERX_BUILD_TESTS=ON
cmake --build build
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
│   ├── core/tracer.h             # Step-by-step instrumentation
│   ├── tensor/                   # Tensor engine (dtype, shape, stride, storage, iterator)
│   ├── nn/
│   │   ├── tokenizer.h           # BPE-style tokenizer
│   │   ├── embedding.h           # Token + positional embedding
│   │   ├── attention.h           # Self-attention (Q, K, V projections)
│   │   ├── matmul.h              # Matrix multiplication
│   │   ├── softmax.h             # Numerically stable softmax
│   │   └── activations.h         # ReLU, GELU
│   ├── memory/                   # Arena allocator
│   ├── kernels/                  # SIMD matmul (ARM NEON)
│   ├── graph/                    # Computational graph
│   ├── parallel/                 # Thread pool
│   └── quantize/                 # INT8 quantization
├── src/wasm/                     # Emscripten bindings
├── tests/                        # 216+ unit tests (GoogleTest)
├── visualizer/
│   ├── src/
│   │   ├── App.jsx               # Router + layout
│   │   ├── components/
│   │   │   ├── Landing.jsx       # Home (4-module path)
│   │   │   ├── MathExplorer.jsx  # Math Lab (6 topics)
│   │   │   ├── TextPipeline.jsx  # Text AI (8 stages)
│   │   │   ├── ImagePipeline.jsx # Vision AI (6 stages)
│   │   │   ├── MNISTLive.jsx     # Draw & predict
│   │   │   └── GuidedTrack.jsx   # Bottom navigation
│   │   └── engine/               # WASM loader + model weights
│   └── public/                   # Logo, compiled WASM
├── benchmarks/                   # Performance benchmarks
└── logo/                         # Brand assets
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Engine | C++20 (concepts, constexpr, ranges) |
| SIMD | ARM NEON (16-byte aligned) |
| WASM | Emscripten + embind |
| Frontend | React 19 + Ant Design + Vite |
| Routing | react-router-dom v7 |
| Build | CMake 3.20+ / Vite 8 |
| Testing | GoogleTest 1.14 |
| Dataset | MNIST (60K training images) |
| Compute | 100% client-side (no server) |
| Target | macOS ARM64, Browser (WASM) |

## Performance

| Metric | Value |
|--------|-------|
| MatMul throughput | 22 GFLOPS (Apple Silicon) |
| Memory allocation | 918x faster than malloc |
| MNIST inference | < 1ms per digit |
| Test coverage | 216+ tests, all passing |
| Bundle size | ~300KB gzipped |

## Roadmap

- [ ] Computational graph with topological execution
- [ ] Full operator kernel library (Add, Sub, Div, Pow, LayerNorm)
- [ ] Thread pool scheduler for parallel execution
- [ ] ONNX model loading
- [ ] Pre-trained text model weights (real next-word predictions)
- [ ] Autograd (backward pass, gradient accumulation)
- [ ] GPU backend (Metal on macOS)

## License

MIT
