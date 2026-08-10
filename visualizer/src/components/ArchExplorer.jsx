import { useState } from "react";
import "./ArchExplorer.css";

const MODULES = [
  {
    id: "tensor",
    label: "Tensor Engine",
    icon: "📐",
    color: "#3b82f6",
    summary: "The foundation — stores N-dimensional data with zero-copy operations",
    details: [
      "Shape: stack-allocated, max rank 8, NumPy-style broadcasting",
      "Storage: 16-byte aligned, RAII ownership, move-only semantics",
      "Stride: row-major layout, contiguity detection",
      "Operations: reshape, slice, transpose — metadata only, no data copy",
    ],
    code: `Tensor<Float32> t({2, 3});  // 2×3 matrix\nt(0, 1) = 3.14f;           // element access\nauto v = t.reshape({6});   // zero-copy view`,
    metrics: { tests: 113, loc: "~1200" },
  },
  {
    id: "kernels",
    label: "SIMD Kernels",
    icon: "⚡",
    color: "#8b5cf6",
    summary: "Hand-tuned ARM NEON matrix multiply — 22 GFLOPS on Apple M1",
    details: [
      "Naive: triple loop baseline (~8 GFLOPS)",
      "Tiled: cache-blocked for L1 (64×64 tiles fit 48KB)",
      "NEON: 4×4 micro-kernel using vfmaq_f32 intrinsics",
      "Dispatch: auto-selects best kernel by matrix dimensions",
    ],
    code: `// 4×4 NEON micro-kernel core:\nfloat32x4_t c0 = vld1q_f32(C);\nfloat32x4_t b  = vld1q_f32(B + k*N);\nc0 = vfmaq_f32(c0, vdupq_n_f32(A[k]), b);`,
    metrics: { gflops: "22", speedup: "2.5×" },
  },
  {
    id: "memory",
    label: "Memory Manager",
    icon: "🧠",
    color: "#22c55e",
    summary: "Arena allocator — 918× faster than malloc for inference workloads",
    details: [
      "Arena: bump pointer, O(1) alloc, instant reset per inference",
      "Buffer Pool: size-bucketed reuse, 100% hit rate after warmup",
      "Aligned Allocator: 64-byte cache-line aligned for NEON",
      "Profiler: peak usage tracking, utilization metrics",
    ],
    code: `Arena arena(16_MB);\nfloat* Q = arena.alloc<float>(seq * dim);  // 4ns\nfloat* K = arena.alloc<float>(seq * dim);  // 4ns\n// ... inference ...\narena.reset();  // instant! all freed`,
    metrics: { speed: "1.3 ns/alloc", vs_malloc: "918×" },
  },
  {
    id: "graph",
    label: "Computational Graph",
    icon: "🔗",
    color: "#f59e0b",
    summary: "DAG executor with operator fusion — eliminates unnecessary memory traffic",
    details: [
      "Operators: MatMul, ReLU, GELU, Softmax, Add (+ fused variants)",
      "Graph: DAG with topological sort (Kahn's algorithm)",
      "Executor: arena-backed, walks topo order, allocates intermediates",
      "Optimizer: dead node elimination + MatMul+ReLU fusion",
    ],
    code: `auto mm = graph.add_node(MatMulOp, {input, weights});\nauto relu = graph.add_node(ReLUOp, {mm});\noptimize(graph);  // → FusedMatMulReLU\n// Saves one full memory pass over M×N elements`,
    metrics: { passes: "3", fusion: "MatMul+Act" },
  },
  {
    id: "quantize",
    label: "INT8 Quantization",
    icon: "📦",
    color: "#ec4899",
    summary: "4× memory compression with <1% accuracy loss — deployment ready",
    details: [
      "Symmetric: scale = max(|x|)/127, zero maps exactly to 0",
      "INT8 MatMul: int8×int8 → int32 accumulate → float dequantize",
      "Dynamic: quantize activations per-inference, weights pre-quantized",
      "Quality: >35 dB SNR for normally distributed weights",
    ],
    code: `auto params = compute_symmetric_params(weights);\nquantize_tensor(weights, q_weights, params);\n// 4× less memory, 2-4× faster compute\nauto result = quantized_matmul(A_q, B_q, M, K, N,\n                               scale_a, scale_b);`,
    metrics: { compression: "4×", snr: ">35 dB" },
  },
  {
    id: "parallel",
    label: "Thread Pool",
    icon: "🔀",
    color: "#06b6d4",
    summary: "Task graph with dependency-driven parallel execution",
    details: [
      "Pool: fixed N workers, condition_variable wake, ~50ns dispatch",
      "TaskGraph: DAG of tasks, auto-submits when deps are met",
      "parallel_for: chunk-based loop splitting across cores",
      "parallel_reduce: per-chunk partial results then combine",
    ],
    code: `ThreadPool pool(8);\nTaskGraph graph;\nauto q = graph.add_task(compute_Q, {input});\nauto k = graph.add_task(compute_K, {input});\nauto v = graph.add_task(compute_V, {input});\ngraph.add_task(attention, {q, k, v}); // waits\ngraph.execute(pool); // Q,K,V run in parallel!`,
    metrics: { dispatch: "~50 ns", pattern: "inter+intra op" },
  },
  {
    id: "inference",
    label: "MNIST Inference",
    icon: "🎯",
    color: "#f97316",
    summary: "End-to-end: trained model → C++ engine → 100% accuracy, 7485 img/sec",
    details: [
      "Model: 784→128(ReLU)→10(Softmax), trained in PyTorch",
      "Export: custom binary format (INFX), 397 KB weights",
      "Inference: arena-backed, zero malloc on hot path",
      "Result: 100/100 correct, 133µs per image",
    ],
    code: `auto model = load_model("mnist_weights.bin");\narena.reset();\nfloat* hidden = arena.alloc<float>(128);\nlinear_forward(W1, bias1, image, hidden, 128, 784);\nrelu_inplace(hidden, 128);\nlinear_forward(W2, bias2, hidden, logits, 10, 128);\n// Prediction: argmax(softmax(logits))`,
    metrics: { accuracy: "100%", throughput: "7485/sec" },
  },
];

const FLOW_CONNECTIONS = [
  { from: "tensor", to: "kernels", label: "Tensor data → SIMD compute" },
  { from: "tensor", to: "memory", label: "Storage backed by allocator" },
  { from: "kernels", to: "graph", label: "Ops use optimized kernels" },
  { from: "memory", to: "graph", label: "Executor uses arena" },
  { from: "graph", to: "quantize", label: "Graph ops can be quantized" },
  { from: "graph", to: "parallel", label: "Independent ops run parallel" },
  { from: "parallel", to: "inference", label: "Pool powers inference" },
  { from: "quantize", to: "inference", label: "INT8 for deployment" },
];

export default function ArchExplorer() {
  const [selected, setSelected] = useState(null);
  const [hoveredConn, setHoveredConn] = useState(null);

  const selectedModule = MODULES.find(m => m.id === selected);

  return (
    <div className="arch">
      <div className="arch-header">
        <h1>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 3v18"/>
          </svg>
          InferX Architecture
        </h1>
        <p>Click any module to explore how it works — from tensors to real inference</p>
      </div>

      {/* System Diagram */}
      <div className="arch-diagram">
        <div className="arch-grid">
          {MODULES.map((mod) => (
            <button
              key={mod.id}
              className={`arch-module ${selected === mod.id ? "selected" : ""}`}
              style={{ "--mod-color": mod.color }}
              onClick={() => setSelected(selected === mod.id ? null : mod.id)}
            >
              <span className="mod-icon">{mod.icon}</span>
              <span className="mod-label">{mod.label}</span>
              <span className="mod-summary">{mod.summary}</span>
            </button>
          ))}
        </div>

        {/* Data Flow */}
        <div className="arch-flow">
          <div className="flow-title">Data Flow</div>
          <div className="flow-connections">
            {FLOW_CONNECTIONS.map((conn, i) => (
              <div
                key={i}
                className={`flow-conn ${hoveredConn === i ? "highlight" : ""}`}
                onMouseEnter={() => setHoveredConn(i)}
                onMouseLeave={() => setHoveredConn(null)}
              >
                <span className="conn-from">{conn.from}</span>
                <span className="conn-arrow">→</span>
                <span className="conn-to">{conn.to}</span>
                <span className="conn-label">{conn.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Detail Panel */}
      {selectedModule && (
        <div className="arch-detail" style={{ "--mod-color": selectedModule.color }}>
          <div className="detail-header">
            <span className="detail-icon">{selectedModule.icon}</span>
            <div>
              <h2>{selectedModule.label}</h2>
              <p>{selectedModule.summary}</p>
            </div>
          </div>

          <div className="detail-body">
            {/* How it works */}
            <div className="detail-section">
              <h3>How It Works</h3>
              <ul className="detail-list">
                {selectedModule.details.map((d, i) => (
                  <li key={i}>{d}</li>
                ))}
              </ul>
            </div>

            {/* Code example */}
            <div className="detail-section">
              <h3>Code Example</h3>
              <pre className="detail-code">
                <code>{selectedModule.code}</code>
              </pre>
            </div>

            {/* Metrics */}
            <div className="detail-section">
              <h3>Key Metrics</h3>
              <div className="detail-metrics">
                {Object.entries(selectedModule.metrics).map(([key, val]) => (
                  <div key={key} className="metric-card">
                    <span className="metric-label">{key}</span>
                    <span className="metric-value">{val}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="arch-legend">
        <div className="legend-title">Project Stats</div>
        <div className="legend-stats">
          <div className="legend-stat">
            <span className="ls-value">216+</span>
            <span className="ls-label">Tests Passing</span>
          </div>
          <div className="legend-stat">
            <span className="ls-value">27</span>
            <span className="ls-label">C++ Headers</span>
          </div>
          <div className="legend-stat">
            <span className="ls-value">7</span>
            <span className="ls-label">Modules</span>
          </div>
          <div className="legend-stat">
            <span className="ls-value">C++20</span>
            <span className="ls-label">Standard</span>
          </div>
          <div className="legend-stat">
            <span className="ls-value">0</span>
            <span className="ls-label">External Deps</span>
          </div>
        </div>
      </div>
    </div>
  );
}
