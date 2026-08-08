import { useState } from "react";
import InteractiveBuilder from "./InteractiveBuilder";
import "./TensorPage.css";

const TENSOR_TYPES = [
  { dims: 0, name: "Scalar", shape: "[]", example: "5.0", desc: "A single number", visual: "scalar", color: "#4a90e2" },
  { dims: 1, name: "Vector", shape: "[4]", example: "[0.2, -0.5, 0.8, 0.1]", desc: "A list of numbers", visual: "vector", color: "#a78bfa" },
  { dims: 2, name: "Matrix", shape: "[3, 4]", example: "3 rows × 4 columns", desc: "A 2D table", visual: "matrix", color: "#34d399" },
  { dims: 3, name: "3D Tensor", shape: "[2, 3, 4]", example: "2 layers of 3×4", desc: "A cube of numbers", visual: "cube", color: "#fb923c" },
  { dims: 4, name: "4D Tensor", shape: "[1, 3, 28, 28]", example: "Image: batch×channels×H×W", desc: "Used for images in AI", visual: "hyper", color: "#f87171" },
];

const AI_EXAMPLES = [
  { context: "Text AI — Token IDs", shape: "[7]", meaning: "7 words in your sentence, each represented as a number", type: "1D Vector" },
  { context: "Text AI — Embeddings", shape: "[7, 768]", meaning: "7 words, each described by 768 numbers (meaning vector)", type: "2D Matrix" },
  { context: "Text AI — Attention Scores", shape: "[7, 7]", meaning: "How much each word attends to every other word", type: "2D Matrix" },
  { context: "Image AI — Input", shape: "[1, 1, 28, 28]", meaning: "1 grayscale image, 28×28 pixels", type: "4D Tensor" },
  { context: "Image AI — After Conv", shape: "[1, 16, 26, 26]", meaning: "16 different pattern detections across the image", type: "4D Tensor" },
  { context: "Image AI — Logits", shape: "[1, 10]", meaning: "Score for each of 10 digit classes", type: "2D Matrix" },
  { context: "GPT-4 — Hidden State", shape: "[1, 2048, 12288]", meaning: "2048 tokens, each with 12,288-dim representation", type: "3D Tensor" },
];

export default function TensorPage() {
  const [showPlayground, setShowPlayground] = useState(false);
  const [selectedType, setSelectedType] = useState(null);

  return (
    <div className="tensor-page">
      <div className="tp-header">
        <h1><span className="tp-icon">▦</span> Tensors — The Building Blocks of AI</h1>
        <p>Every number flowing through a neural network lives inside a tensor. They're the fundamental data structure of all AI.</p>
      </div>

      <div className="demo-banner">
        <span className="demo-banner-icon">💡</span>
        <div className="demo-banner-text">
          <strong>Tensors are not a separate concept from Text AI or Image AI</strong> — they're the container that holds ALL the data at every step. Token IDs? A tensor. Pixel values? A tensor. Attention scores? A tensor. Everything is tensors.
        </div>
      </div>

      {/* What is a Tensor */}
      <section className="ts-section">
        <h2>What is a Tensor?</h2>
        <p className="ts-desc">A tensor is simply a <strong>multi-dimensional array of numbers</strong>. Different dimensions are used for different types of data:</p>

        <div className="ts-types-grid">
          {TENSOR_TYPES.map((t) => (
            <div
              key={t.dims}
              className={`ts-type-card ${selectedType === t.dims ? "active" : ""}`}
              style={{ "--accent-color": t.color }}
              onClick={() => setSelectedType(selectedType === t.dims ? null : t.dims)}
            >
              <div className="ts-type-visual">
                <TensorMiniVis type={t.visual} color={t.color} />
              </div>
              <div className="ts-type-info">
                <span className="ts-type-name">{t.name}</span>
                <span className="ts-type-dims">{t.dims}D</span>
              </div>
              <code className="ts-type-shape">{t.shape}</code>
              <span className="ts-type-desc">{t.desc}</span>
            </div>
          ))}
        </div>

        {selectedType !== null && (
          <div className="ts-type-detail">
            <div className="ts-detail-header">
              <strong>{TENSOR_TYPES[selectedType].name}</strong> — {TENSOR_TYPES[selectedType].desc}
            </div>
            <div className="ts-detail-body">
              <p><strong>Shape:</strong> <code>{TENSOR_TYPES[selectedType].shape}</code></p>
              <p><strong>Example:</strong> {TENSOR_TYPES[selectedType].example}</p>
              <p><strong>In AI:</strong> {
                selectedType === 0 ? "A loss value, a learning rate, a single prediction confidence" :
                selectedType === 1 ? "Token IDs [7], bias terms [768], output logits [10]" :
                selectedType === 2 ? "Embedding matrix [50000×768], attention scores [7×7], weight matrices" :
                selectedType === 3 ? "Batch of sequences [32, 128, 768] — 32 sentences, 128 tokens each, 768 dims" :
                "Images [batch, channels, height, width] — the standard format for CNNs"
              }</p>
            </div>
          </div>
        )}
      </section>

      {/* Where tensors appear in AI */}
      <section className="ts-section">
        <h2>Tensors in Your AI Pipelines</h2>
        <p className="ts-desc">Here's exactly where tensors appear in the Text AI and Image AI pages you've already explored:</p>

        <div className="ts-examples-table">
          <div className="ts-ex-header">
            <span>Where</span><span>Shape</span><span>What it means</span>
          </div>
          {AI_EXAMPLES.map((ex, i) => (
            <div key={i} className="ts-ex-row">
              <span className="ts-ex-context">{ex.context}</span>
              <code className="ts-ex-shape">{ex.shape}</code>
              <span className="ts-ex-meaning">{ex.meaning}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Key operations */}
      <section className="ts-section">
        <h2>Key Tensor Operations</h2>
        <p className="ts-desc">Every AI model repeatedly applies these operations to tensors:</p>

        <div className="ts-ops-grid">
          <div className="ts-op-card">
            <span className="ts-op-icon">✖️</span>
            <strong>Matrix Multiply</strong>
            <p>[7×768] × [768×768] = [7×768]</p>
            <small>Used in: attention, linear layers, embeddings</small>
          </div>
          <div className="ts-op-card">
            <span className="ts-op-icon">🔄</span>
            <strong>Reshape</strong>
            <p>[1, 16, 13, 13] → [1, 2704]</p>
            <small>Used in: flatten before FC layers</small>
          </div>
          <div className="ts-op-card">
            <span className="ts-op-icon">✂️</span>
            <strong>Slice</strong>
            <p>[7, 768] → [1, 768] (last token)</p>
            <small>Used in: getting output for prediction</small>
          </div>
          <div className="ts-op-card">
            <span className="ts-op-icon">🔀</span>
            <strong>Transpose</strong>
            <p>[7, 768] → [768, 7]</p>
            <small>Used in: attention Q×K^T</small>
          </div>
          <div className="ts-op-card">
            <span className="ts-op-icon">➕</span>
            <strong>Element-wise Add</strong>
            <p>[7, 768] + [7, 768] = [7, 768]</p>
            <small>Used in: residual connections, positional encoding</small>
          </div>
          <div className="ts-op-card">
            <span className="ts-op-icon">📡</span>
            <strong>Broadcast</strong>
            <p>[7, 1] + [1, 768] = [7, 768]</p>
            <small>Used in: adding bias, scaling</small>
          </div>
        </div>
      </section>

      {/* Playground toggle */}
      <section className="ts-section">
        <div className="ts-playground-header">
          <h2>Tensor Playground</h2>
          <button className="btn btn-primary" onClick={() => setShowPlayground(!showPlayground)}>
            {showPlayground ? "Hide Playground" : "Open Interactive Playground"}
          </button>
        </div>
        <p className="ts-desc">Create tensors, apply operations, and watch step-by-step how the C++ engine processes them. Powered by real compiled C++ via WebAssembly.</p>

        {showPlayground && (
          <div className="ts-playground-wrap">
            <InteractiveBuilder />
          </div>
        )}
      </section>
    </div>
  );
}

// ─── Mini Tensor Visualizations ──────────────────────────────────────
function TensorMiniVis({ type, color }) {
  switch (type) {
    case "scalar":
      return (
        <div className="ts-mini-vis">
          <div className="ts-scalar" style={{ background: color }}>5.0</div>
        </div>
      );
    case "vector":
      return (
        <div className="ts-mini-vis">
          <div className="ts-vector">
            {[0.2, -0.5, 0.8, 0.1].map((v, i) => (
              <div key={i} className="ts-v-cell" style={{ background: `${color}${Math.round((Math.abs(v)+0.2)*200).toString(16).padStart(2,'0')}` }} />
            ))}
          </div>
        </div>
      );
    case "matrix":
      return (
        <div className="ts-mini-vis">
          <div className="ts-matrix">
            {Array.from({ length: 12 }, (_, i) => (
              <div key={i} className="ts-m-cell" style={{ background: `${color}${Math.round((((i*37)%100)/100)*200+55).toString(16).padStart(2,'0')}` }} />
            ))}
          </div>
        </div>
      );
    case "cube":
      return (
        <div className="ts-mini-vis">
          <div className="ts-cube">
            <div className="ts-cube-face ts-cube-front" style={{ borderColor: color }} />
            <div className="ts-cube-face ts-cube-back" style={{ borderColor: color }} />
          </div>
        </div>
      );
    case "hyper":
      return (
        <div className="ts-mini-vis">
          <div className="ts-hyper">
            {[0,1,2].map(i => (
              <div key={i} className="ts-hyper-layer" style={{ borderColor: color, opacity: 1 - i * 0.25 }} />
            ))}
          </div>
        </div>
      );
    default:
      return null;
  }
}
