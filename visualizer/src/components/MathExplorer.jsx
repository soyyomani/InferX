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
  },
  {
    id: "softmax",
    name: "Softmax",
    icon: "σ",
    desc: "Convert scores to probabilities",
    color: "purple",
  },
  {
    id: "relu",
    name: "ReLU",
    icon: "⌐",
    desc: "max(0, x) — the simplest activation",
    color: "success",
  },
  {
    id: "gelu",
    name: "GELU",
    icon: "≈",
    desc: "Smooth activation used in GPT/BERT",
    color: "orange",
  },
  {
    id: "attention",
    name: "Attention",
    icon: "◎",
    desc: "How tokens attend to each other",
    color: "cyan",
  },
  {
    id: "tokenize",
    name: "Tokenizer",
    icon: "✂",
    desc: "Text → number sequences",
    color: "warning",
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

      {/* Input Panel */}
      {selectedOp && (
        <div className="op-input-panel card">
          <h3>{OPERATIONS.find((o) => o.id === selectedOp)?.name}</h3>
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
