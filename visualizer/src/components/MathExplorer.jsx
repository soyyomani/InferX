import { useState } from "react";
import {
  traceMatMul,
  traceSoftmax,
  traceReLU,
  traceGELU,
  traceTokenize,
  traceEmbedding,
  traceAttention,
} from "../engine/nn_wasm";
import StepViewer from "./StepViewer";
import {
  AttentionHeatmap,
  MatMulAnimation,
  SoftmaxChart,
  ActivationPlot,
} from "./visualizations";
import "./MathExplorer.css";

const OPERATIONS = [
  {
    id: "matmul",
    name: "Matrix Multiply",
    icon: "×",
    desc: "The fundamental operation of neural networks",
    color: "accent",
    overview: {
      definition: "Matrix multiplication (MatMul) takes two matrices and produces a third by computing dot products of rows and columns. It's the single most important operation in deep learning — every linear layer, attention mechanism, and embedding lookup is a MatMul.",
      formula: "C[i][j] = Σₖ A[i][k] × B[k][j]",
      intuition: "Think of it as 'mixing inputs with learned weights.' Each output number is a weighted combination of all inputs — the weights determine what the network has learned.",
      usedIn: ["Every linear/dense layer", "Attention (Q×K, attn×V)", "Embedding lookup", "Convolution (im2col)"],
      complexity: "O(M × K × N) — for [M×K] × [K×N]",
      realWorld: "GPT-4 does ~1.8 trillion MatMul operations per token generated.",
    },
  },
  {
    id: "softmax",
    name: "Softmax",
    icon: "σ",
    desc: "Convert scores to probabilities",
    color: "purple",
    overview: {
      definition: "Softmax converts a vector of arbitrary real numbers (logits) into a probability distribution — all values become positive and sum to exactly 1.0. It amplifies differences: larger inputs get disproportionately larger probabilities.",
      formula: "softmax(xᵢ) = e^(xᵢ - max) / Σⱼ e^(xⱼ - max)",
      intuition: "It's like a 'winner-take-more' function. If one score is slightly higher, softmax makes it much more dominant. Subtracting max prevents numerical overflow.",
      usedIn: ["Attention weights (Q·K → probabilities)", "Final output layer (logits → next-word probabilities)", "Classification (scores → class probabilities)"],
      complexity: "O(n) — two passes: one for max/exp, one for normalize",
      realWorld: "Every time ChatGPT picks a word, it runs softmax over 100K+ vocabulary scores.",
    },
  },
  {
    id: "relu",
    name: "ReLU",
    icon: "⌐",
    desc: "max(0, x) — the simplest activation",
    color: "success",
    overview: {
      definition: "ReLU (Rectified Linear Unit) is the simplest non-linear activation function: it passes positive values unchanged and replaces negatives with zero. Without non-linearity, stacking layers would be pointless (multiple linear layers = one linear layer).",
      formula: "ReLU(x) = max(0, x)",
      intuition: "Think of it as a gate — it lets 'excited' neurons pass through and silences the rest. This creates sparsity (many zeros) which helps the network learn distinct features.",
      usedIn: ["Hidden layers in CNNs", "Feed-forward networks in older transformers", "Most computer vision models"],
      complexity: "O(n) — single comparison per element",
      realWorld: "ResNet-50 applies ReLU ~23 million times per forward pass on a single image.",
    },
  },
  {
    id: "gelu",
    name: "GELU",
    icon: "≈",
    desc: "Smooth activation used in GPT/BERT",
    color: "orange",
    overview: {
      definition: "GELU (Gaussian Error Linear Unit) is a smooth, differentiable activation that approximates ReLU but doesn't have the hard cutoff at zero. Small negative values get slightly reduced instead of killed entirely.",
      formula: "GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))",
      intuition: "Unlike ReLU which is binary (pass or kill), GELU is probabilistic — it weights values by 'how likely they are to be positive.' This gives smoother gradients and slightly better training dynamics.",
      usedIn: ["GPT-2, GPT-3, GPT-4 (all feed-forward layers)", "BERT", "Most modern transformer models"],
      complexity: "O(n) — more expensive than ReLU (requires tanh), but still element-wise",
      realWorld: "GPT-4 uses GELU in every one of its 96 transformer layers × 2 FFN sublayers = 192 GELU applications per token.",
    },
  },
  {
    id: "attention",
    name: "Attention",
    icon: "◎",
    desc: "How tokens attend to each other",
    color: "cyan",
    overview: {
      definition: "Self-Attention is the core innovation of transformers. Each token looks at EVERY other token and computes a relevance score, then uses those scores to create a context-aware representation. This is how 'bank' knows if it means 'river bank' or 'money bank.'",
      formula: "Attention(Q,K,V) = softmax(Q×Kᵀ / √d_k) × V",
      intuition: "Each word asks 'what should I pay attention to?' (Query), advertises 'what do I contain?' (Key), and offers 'here's my info' (Value). Dot product of Q and K = relevance score. High scores mean strong connection.",
      usedIn: ["Every transformer layer (GPT, BERT, Claude)", "Cross-attention in translation", "Vision Transformers (ViT)", "Diffusion models (Stable Diffusion)"],
      complexity: "O(n² × d) — quadratic in sequence length (why context windows are expensive)",
      realWorld: "Claude's 200K context window means attention computes 200,000 × 200,000 = 40 billion score pairs per layer.",
    },
  },
  {
    id: "tokenize",
    name: "Tokenizer",
    icon: "✂",
    desc: "Text → number sequences",
    color: "warning",
    overview: {
      definition: "Tokenization is the first step in any NLP pipeline — it converts raw text into a sequence of integer IDs that the neural network can process. Each ID maps to a token (word, subword, or character) in a fixed vocabulary.",
      formula: "\"Hello world\" → split → [\"Hello\", \"world\"] → lookup → [15496, 995]",
      intuition: "AI can't read text directly. Tokenization is like giving each word a number name. The model then works entirely with these numbers — it never sees the actual letters.",
      usedIn: ["Input preprocessing for all language models", "BPE (GPT), WordPiece (BERT), SentencePiece (T5)", "Code models (special tokens for syntax)"],
      complexity: "O(n × m) — n = text length, m = max token length for BPE matching",
      realWorld: "GPT-4's tokenizer has ~100K tokens. 'Hello' = 1 token, but 'tokenization' might be 3 tokens: 'token' + 'ization' + boundary.",
    },
  },
];

function MatMulInput({ onRun }) {
  const [M, setM] = useState(2);
  const [K, setK] = useState(3);
  const [N, setN] = useState(2);

  function run() {
    const A = Array.from({ length: M * K }, () => parseFloat((Math.random() * 2 - 1).toFixed(3)));
    const B = Array.from({ length: K * N }, () => parseFloat((Math.random() * 2 - 1).toFixed(3)));
    onRun(traceMatMul(A, M, K, B, K, N), { A, B, M, K, N });
  }

  return (
    <div className="op-inputs">
      <div className="op-param-row">
        <label>A rows (M):</label>
        <input type="number" min={1} max={8} value={M} onChange={(e) => setM(+e.target.value)} />
        <label>Inner (K):</label>
        <input type="number" min={1} max={8} value={K} onChange={(e) => setK(+e.target.value)} />
        <label>B cols (N):</label>
        <input type="number" min={1} max={8} value={N} onChange={(e) => setN(+e.target.value)} />
      </div>
      <div className="op-shape-preview">
        A[{M}×{K}] × B[{K}×{N}] = C[{M}×{N}]
      </div>
      <button className="btn btn-primary" onClick={run}>Compute MatMul</button>
    </div>
  );
}

function SoftmaxInput({ onRun }) {
  const [values, setValues] = useState("2.0, 1.0, 0.1, -1.0, 3.0");

  function run() {
    const logits = values.split(",").map((v) => parseFloat(v.trim())).filter((v) => !isNaN(v));
    if (logits.length > 0) onRun(traceSoftmax(logits));
  }

  return (
    <div className="op-inputs">
      <label>Logits (comma-separated):</label>
      <input
        type="text"
        className="input-field"
        value={values}
        onChange={(e) => setValues(e.target.value)}
        placeholder="2.0, 1.0, 0.1, -1.0, 3.0"
      />
      <div className="op-presets">
        <span>Presets:</span>
        <button onClick={() => setValues("2.0, 1.0, 0.1")}>Small (3)</button>
        <button onClick={() => setValues("5.0, 2.0, 1.0, -1.0, 0.5, -2.0, 3.0")}>Medium (7)</button>
        <button onClick={() => setValues("10.0, 1.0, 0.1, -5.0, 0.0")}>Confident</button>
        <button onClick={() => setValues("1.0, 1.1, 0.9, 1.0, 1.05")}>Uncertain</button>
      </div>
      <button className="btn btn-primary" onClick={run}>Compute Softmax</button>
    </div>
  );
}

function ReLUInput({ onRun }) {
  const [values, setValues] = useState("-2.0, -0.5, 0.0, 0.3, 1.5, -1.2, 2.1, -0.1");

  function run() {
    const input = values.split(",").map((v) => parseFloat(v.trim())).filter((v) => !isNaN(v));
    if (input.length > 0) onRun(traceReLU(input));
  }

  return (
    <div className="op-inputs">
      <label>Input values (comma-separated):</label>
      <input
        type="text"
        className="input-field"
        value={values}
        onChange={(e) => setValues(e.target.value)}
      />
      <button className="btn btn-primary" onClick={run}>Apply ReLU</button>
    </div>
  );
}

function GELUInput({ onRun }) {
  const [values, setValues] = useState("-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0");

  function run() {
    const input = values.split(",").map((v) => parseFloat(v.trim())).filter((v) => !isNaN(v));
    if (input.length > 0) onRun(traceGELU(input));
  }

  return (
    <div className="op-inputs">
      <label>Input values (comma-separated):</label>
      <input
        type="text"
        className="input-field"
        value={values}
        onChange={(e) => setValues(e.target.value)}
      />
      <button className="btn btn-primary" onClick={run}>Apply GELU</button>
    </div>
  );
}

function AttentionInput({ onRun }) {
  const [seqLen, setSeqLen] = useState(4);

  function run() {
    onRun(traceAttention(seqLen));
  }

  return (
    <div className="op-inputs">
      <div className="op-param-row">
        <label>Sequence length:</label>
        <input
          type="number"
          min={2}
          max={8}
          value={seqLen}
          onChange={(e) => setSeqLen(+e.target.value)}
        />
      </div>
      <p className="op-hint">Simulates {seqLen} tokens attending to each other with embed_dim=32</p>
      <button className="btn btn-primary" onClick={run}>Run Attention</button>
    </div>
  );
}

function TokenizeInput({ onRun }) {
  const [text, setText] = useState("Hello world");

  function run() {
    onRun(traceTokenize(text));
  }

  return (
    <div className="op-inputs">
      <label>Text to tokenize:</label>
      <input
        type="text"
        className="input-field"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Hello world"
      />
      <button className="btn btn-primary" onClick={run}>Tokenize</button>
    </div>
  );
}

export default function MathExplorer() {
  const [selectedOp, setSelectedOp] = useState(null);
  const [traceSteps, setTraceSteps] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [matData, setMatData] = useState(null);

  function handleResult(steps, extra) {
    setTraceSteps(steps);
    setCurrentStep(0);
    if (extra) setMatData(extra);
  }

  function renderInputPanel() {
    switch (selectedOp) {
      case "matmul": return <MatMulInput onRun={(s, d) => handleResult(s, d)} />;
      case "softmax": return <SoftmaxInput onRun={handleResult} />;
      case "relu": return <ReLUInput onRun={handleResult} />;
      case "gelu": return <GELUInput onRun={handleResult} />;
      case "attention": return <AttentionInput onRun={handleResult} />;
      case "tokenize": return <TokenizeInput onRun={handleResult} />;
      default: return null;
    }
  }

  return (
    <div className="math-explorer animate-in">
      <div className="page-header">
        <h1 className="page-title">
          <span className="page-icon">∑</span>
          Math Lab
        </h1>
        <p className="page-desc">
          Explore individual AI math operations interactively.
          Pick an operation, set inputs, and watch the step-by-step computation.
        </p>
      </div>

      {/* Operation Selector */}
      <div className="op-grid">
        {OPERATIONS.map((op) => (
          <button
            key={op.id}
            className={`op-card op-${op.color} ${selectedOp === op.id ? "active" : ""}`}
            onClick={() => { setSelectedOp(op.id); setTraceSteps([]); }}
          >
            <span className="op-icon">{op.icon}</span>
            <span className="op-name">{op.name}</span>
            <span className="op-desc">{op.desc}</span>
          </button>
        ))}
      </div>

      {/* Overview Panel — shows educational content when card is tapped */}
      {selectedOp && <OperationOverview op={OPERATIONS.find((o) => o.id === selectedOp)} />}

      {/* Input Panel */}
      {selectedOp && (
        <div className="op-input-panel card">
          <h3>Try it yourself</h3>
          {renderInputPanel()}
        </div>
      )}

      {/* Matrix visualization for matmul */}
      {selectedOp === "matmul" && matData && (
        <div className="matrix-vis">
          <MatrixDisplay data={matData.A} rows={matData.M} cols={matData.K} label="A" />
          <span className="matrix-op-symbol">×</span>
          <MatrixDisplay data={matData.B} rows={matData.K} cols={matData.N} label="B" />
        </div>
      )}

      {/* Animated Visualizations */}
      {selectedOp === "matmul" && <MatMulAnimation M={3} K={3} N={3} />}
      {selectedOp === "softmax" && <SoftmaxChart logits={[2.0, 1.0, 0.1, -1.0, 3.0]} />}
      {selectedOp === "relu" && <ActivationPlot type="relu" />}
      {selectedOp === "gelu" && <ActivationPlot type="gelu" />}
      {selectedOp === "attention" && <AttentionHeatmap tokens={["How", "does", "AI", "work"]} />}

      {/* Results */}
      {traceSteps.length > 0 && (
        <StepViewer
          steps={traceSteps}
          currentStep={currentStep}
          onStepChange={setCurrentStep}
          title={OPERATIONS.find((o) => o.id === selectedOp)?.name || ""}
        />
      )}
    </div>
  );
}

function MatrixDisplay({ data, rows, cols, label }) {
  return (
    <div className="matrix-display">
      <span className="matrix-label">{label} [{rows}×{cols}]</span>
      <div className="matrix-grid" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
        {data.map((v, i) => (
          <div key={i} className="matrix-cell">
            {typeof v === "number" ? v.toFixed(2) : v}
          </div>
        ))}
      </div>
    </div>
  );
}

function OperationOverview({ op }) {
  if (!op || !op.overview) return null;
  const { overview, icon, name, color } = op;

  const iconBgClass = `op-${color}`;

  return (
    <div className="op-overview">
      <div className="op-overview-header">
        <div className={`op-overview-icon op-icon ${iconBgClass}`} style={{ width: 40, height: 40, fontSize: '1.2rem' }}>
          {icon}
        </div>
        <div>
          <div className="op-overview-title">{name}</div>
          <div className="op-overview-subtitle">{op.desc}</div>
        </div>
      </div>

      <div className="op-overview-grid">
        <div className="op-overview-item">
          <div className="op-overview-item-label">📖 What is it?</div>
          <div className="op-overview-item-content">{overview.definition}</div>
        </div>

        <div className="op-overview-item">
          <div className="op-overview-item-label">📐 Formula</div>
          <div className="op-overview-item-content">
            <code>{overview.formula}</code>
          </div>
        </div>

        <div className="op-overview-item">
          <div className="op-overview-item-label">💡 Intuition</div>
          <div className="op-overview-item-content">{overview.intuition}</div>
        </div>

        <div className="op-overview-item">
          <div className="op-overview-item-label">🔧 Used in</div>
          <div className="op-overview-item-content">
            <ul>
              {overview.usedIn.map((item, i) => <li key={i}>{item}</li>)}
            </ul>
          </div>
        </div>

        <div className="op-overview-item">
          <div className="op-overview-item-label">⚡ Complexity</div>
          <div className="op-overview-item-content">{overview.complexity}</div>
        </div>

        <div className="op-overview-item">
          <div className="op-overview-item-label">🌍 Real-world scale</div>
          <div className="op-overview-item-content">{overview.realWorld}</div>
        </div>
      </div>
    </div>
  );
}
