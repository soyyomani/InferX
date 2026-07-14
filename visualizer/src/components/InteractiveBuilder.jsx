import { useState, useEffect } from "react";
import {
  initWasm,
  isReady,
  traceFullCreate,
  traceAccess,
  traceReshape,
  traceSlice,
  traceTranspose,
  traceBroadcast,
  traceIterator,
  traceContiguous,
  traceClone,
} from "../engine/wasm";
import TensorGrid3D from "./TensorGrid3D";

// Visual renderers for each component type

function DTypeVisual({ step }) {
  const info = step.internal;
  return (
    <div className="visual-card dtype-card">
      <div className="visual-title">Data Type Resolution</div>
      <div className="dtype-visual-box">
        <div className="dtype-enum-box">
          <span className="enum-label">enum DType</span>
          <div className="enum-values">
            {["Float32", "Float16", "Int8", "Int32", "Int64", "Bool"].map(t => (
              <span key={t} className={`enum-val ${step.title.includes(t) ? "active" : ""}`}>{t}</span>
            ))}
          </div>
        </div>
        <div className="arrow-down">↓</div>
        <div className="traits-box">
          <div className="traits-title">DTypeTraits&lt;{step.title.split("::")[1]?.split(" ")[0] || "Float32"}&gt;</div>
          <div className="traits-grid">
            {info.slice(1, 5).map((line, i) => (
              <div key={i} className="trait-item">{line}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ShapeVisual({ step }) {
  // Parse shape from title like "Shape({2, 3, 4})"
  const match = step.title.match(/\{(.+)\}/);
  const dims = match ? match[1].split(",").map(s => parseInt(s.trim())) : [];
  const numel = dims.reduce((a, b) => a * b, 1);

  return (
    <div className="visual-card shape-card">
      <div className="visual-title">Shape Construction</div>
      <div className="shape-visual-box">
        <div className="shape-array">
          <span className="array-label">std::array&lt;int64_t, 8&gt;</span>
          <div className="array-cells">
            {Array.from({ length: 8 }, (_, i) => (
              <div key={i} className={`array-cell ${i < dims.length ? "filled" : "empty"}`}>
                {i < dims.length ? dims[i] : "—"}
              </div>
            ))}
          </div>
          <div className="array-indices">
            {Array.from({ length: 8 }, (_, i) => (
              <div key={i} className="array-idx">{i}</div>
            ))}
          </div>
        </div>

        <div className="shape-checks">
          <div className="check-item ok">rank = {dims.length} ≤ 8 ✓</div>
          {dims.map((d, i) => (
            <div key={i} className={`check-item ${d > 0 ? "ok" : "fail"}`}>
              dims[{i}] = {d} &gt; 0 {d > 0 ? "✓" : "✗"}
            </div>
          ))}
          <div className="check-item result">
            numel = {dims.join(" × ")} = {numel}
          </div>
        </div>

        {dims.length >= 2 && (
          <div className="shape-grid-visual">
            <span className="grid-label">Visual: [{dims.join("×")}]</span>
            <div className="mini-grid" style={{
              gridTemplateColumns: `repeat(${dims[dims.length - 1]}, 1fr)`,
              gridTemplateRows: `repeat(${Math.min(dims[dims.length - 2] || 1, 8)}, 1fr)`
            }}>
              {Array.from({ length: Math.min((dims[dims.length - 2] || 1) * dims[dims.length - 1], 64) }, (_, i) => (
                <div key={i} className="grid-cell"></div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StrideVisual({ step }) {
  // Parse strides from title
  const match = step.title.match(/\[(.+)\]/);
  const strides = match ? match[1].split(",").map(s => parseInt(s.trim())) : [];

  // Try to infer shape from strides (stride[last]=1, stride[i]=stride[i+1]*shape[i+1])
  // We'll reverse-engineer: shape[last] = stride[last-1] (if exists), etc.
  // Simpler: use strides to guess shape for 3D view
  const inferredShape = [];
  for (let i = 0; i < strides.length; i++) {
    if (i < strides.length - 1) {
      inferredShape.push(Math.round(strides[i] / strides[i + 1]));
    } else {
      // Last dim: guess from context or use a default
      inferredShape.push(strides.length > 1 ? strides[strides.length - 2] : 4);
    }
  }

  return (
    <div className="visual-card stride-card">
      <div className="visual-title">Stride Computation (Row-Major)</div>
      <div className="stride-visual-box">
        <div className="stride-formula">
          <div className="formula-title">Algorithm: work backwards from last dim</div>
          <div className="formula-steps">
            {step.internal.filter(l => l.startsWith("  stride")).map((line, i) => (
              <div key={i} className="formula-step">{line}</div>
            ))}
          </div>
        </div>

        <div className="stride-result">
          <div className="stride-array">
            {strides.map((s, i) => (
              <div key={i} className="stride-cell">
                <div className="stride-val">{s}</div>
                <div className="stride-label">dim {i}</div>
                <div className="stride-meaning">skip {s} elements</div>
              </div>
            ))}
          </div>
        </div>

        <div className="stride-explanation">
          Moving one step in dim 0 skips {strides[0] || "?"} elements in memory.
          Moving one step in the last dim skips 1 element (contiguous).
        </div>
      </div>

      {strides.length >= 1 && strides.length <= 3 && (
        <TensorGrid3D shape={inferredShape} strides={strides} />
      )}
    </div>
  );
}

function StorageVisual({ step }) {
  return (
    <div className="visual-card storage-card">
      <div className="visual-title">Memory Allocation</div>
      <div className="storage-visual-box">
        <div className="memory-block">
          <div className="block-header">TensorStorage</div>
          <div className="block-bar">
            <div className="bar-section allocated">
              <span>Allocated (zeroed)</span>
            </div>
          </div>
          <div className="block-details">
            {step.internal.map((line, i) => (
              <div key={i} className="block-detail">{line}</div>
            ))}
          </div>
        </div>
        <div className="alignment-visual">
          <span className="align-badge">16-byte aligned</span>
          <span className="align-note">Ready for ARM NEON SIMD (128-bit loads)</span>
        </div>
      </div>
    </div>
  );
}

function TensorVisual({ step }) {
  return (
    <div className="visual-card tensor-card">
      <div className="visual-title">Tensor Object Assembled</div>
      <div className="tensor-visual-box">
        <div className="tensor-struct">
          <div className="struct-title">struct Tensor&lt;D&gt;</div>
          <div className="struct-fields">
            {step.internal.map((line, i) => (
              <div key={i} className="struct-field">{line}</div>
            ))}
          </div>
        </div>
        <div className="tensor-ready-badge">Ready to use</div>
      </div>
    </div>
  );
}

function AccessVisual({ step }) {
  return (
    <div className="visual-card access-card">
      <div className="visual-title">Element Access</div>
      <div className="access-visual-box">
        <div className="access-formula">
          {step.internal.map((line, i) => (
            <div key={i} className={`access-line ${line.includes("flat offset") ? "highlight" : ""} ${line === "" ? "spacer" : ""}`}>
              {line}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function OperationVisual({ step }) {
  const isZeroCopy = step.detail.toLowerCase().includes("zero-copy") ||
                     step.detail.toLowerCase().includes("metadata") ||
                     step.internal.some(l => l.includes("NO") && l.includes("copy"));
  return (
    <div className="visual-card operation-card">
      <div className="visual-title">{step.component}</div>
      <div className="operation-visual-box">
        <div className="op-details">
          {step.internal.map((line, i) => (
            <div key={i} className={`op-line ${line === "" ? "spacer" : ""} ${line.includes("NO") ? "highlight-green" : ""}`}>
              {line}
            </div>
          ))}
        </div>
        {isZeroCopy && (
          <div className="zero-copy-badge">Zero-Copy Operation — No allocation, no data movement</div>
        )}
        {!isZeroCopy && step.internal.some(l => l.includes("copy") || l.includes("allocat")) && (
          <div className="copy-badge">Data Copy — New memory allocated</div>
        )}
      </div>
    </div>
  );
}

// Pick the right visual renderer based on component type
function StepVisual({ step, currentShape }) {
  switch (step.component) {
    case "DType": return <DTypeVisual step={step} />;
    case "Shape": return <ShapeVisual step={step} />;
    case "Stride": return <StrideVisual step={step} />;
    case "TensorStorage": return <StorageVisual step={step} />;
    case "Tensor": return <TensorVisual step={step} />;
    case "Tensor::operator()": return <AccessVisual step={step} />;
    default: return <OperationVisual step={step} />;
  }
}

// Real-world examples
const examples = [
  { name: "Grayscale image (8x8)", shape: "8,8", dtype: "Float32", fill: "zeros",
    explanation: "An 8x8 grayscale image. Each pixel is a float [0-1]. Shape [Height, Width]." },
  { name: "RGB image (3x4x4)", shape: "3,4,4", dtype: "Float32", fill: "zeros",
    explanation: "A 4x4 RGB image. Shape [Channels, Height, Width]. 3 channels = R, G, B." },
  { name: "Sentence embedding (1x5x4)", shape: "1,5,4", dtype: "Float32", fill: "ones",
    explanation: "Batch=1, 5 tokens, 4-dim embedding. Like encoding 'hello world is cool yeah'." },
  { name: "Batch of vectors (4x8)", shape: "4,8", dtype: "Float32", fill: "ones",
    explanation: "4 samples, 8 features each. A mini-batch in a neural network." },
  { name: "Weight matrix (3x3)", shape: "3,3", dtype: "Float32", fill: "ones",
    explanation: "A 3x3 weight matrix for a linear layer." },
];

export default function InteractiveBuilder() {
  const [shape, setShape] = useState("2,3,4");
  const [dtype, setDtype] = useState("Float32");
  const [fill, setFill] = useState("zeros");
  const [ops, setOps] = useState([]);
  const [trace, setTrace] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [explanation, setExplanation] = useState("");
  const [running, setRunning] = useState(false);
  const [wasmLoaded, setWasmLoaded] = useState(false);

  const [opType, setOpType] = useState("access");
  const [opParam, setOpParam] = useState("0,1,2");

  useEffect(() => {
    initWasm().then(() => setWasmLoaded(true)).catch(console.error);
  }, []);

  function parseDims(str) {
    return str.split(",").map(Number).filter(n => !isNaN(n) && n > 0);
  }

  function runPipeline() {
    if (!wasmLoaded) return;
    const dims = parseDims(shape);
    if (dims.length === 0) return;

    let steps = [];
    steps.push(...traceFullCreate(dtype, dims, fill));

    let currentShape = dims;
    for (const op of ops) {
      try {
        switch (op.type) {
          case "access": {
            steps.push(...traceAccess(currentShape, parseDims(op.param)));
            break;
          }
          case "reshape": {
            const newShape = parseDims(op.param);
            steps.push(...traceReshape(currentShape, newShape));
            currentShape = newShape;
            break;
          }
          case "slice": {
            const [dim, start, end] = op.param.split(",").map(Number);
            steps.push(...traceSlice(currentShape, dim, start, end));
            const newDims = [...currentShape];
            const cs = Math.max(0, Math.min(start, currentShape[dim]));
            const ce = Math.max(cs, Math.min(end, currentShape[dim]));
            newDims[dim] = ce - cs;
            currentShape = newDims;
            break;
          }
          case "transpose": {
            const [d0, d1] = op.param.split(",").map(Number);
            steps.push(...traceTranspose(currentShape, d0, d1));
            const newDims = [...currentShape];
            [newDims[d0], newDims[d1]] = [newDims[d1], newDims[d0]];
            currentShape = newDims;
            break;
          }
          case "broadcast": {
            steps.push(...traceBroadcast(currentShape, parseDims(op.param)));
            break;
          }
          case "iterate": {
            steps.push(...traceIterator(currentShape, false));
            break;
          }
          case "contiguous": {
            steps.push(...traceContiguous(currentShape));
            break;
          }
          case "clone": {
            steps.push(...traceClone(currentShape));
            break;
          }
        }
      } catch (e) {
        steps.push({ component: "Error", title: e.message, detail: "", internal: [e.message] });
      }
    }

    setTrace(steps);
    setCurrentStep(0);
    setRunning(true);
  }

  function nextStep() {
    if (currentStep < trace.length - 1) setCurrentStep(currentStep + 1);
  }
  function prevStep() {
    if (currentStep > 0) setCurrentStep(currentStep - 1);
  }
  function showAll() {
    setCurrentStep(trace.length - 1);
  }

  function addOperation() {
    setOps([...ops, { type: opType, param: opParam }]);
  }
  function removeOp(index) {
    setOps(ops.filter((_, i) => i !== index));
  }
  function loadExample(ex) {
    setShape(ex.shape);
    setDtype(ex.dtype);
    setFill(ex.fill);
    setExplanation(ex.explanation);
    setOps([]);
    setTrace([]);
    setRunning(false);
  }

  const opPlaceholders = {
    access: "0,1,2 (indices)",
    reshape: "6,4 (new shape)",
    slice: "0,1,3 (dim,start,end)",
    transpose: "0,1 (dim0,dim1)",
    broadcast: "1,4 (shape B)",
    iterate: "",
    contiguous: "",
    clone: "",
  };

  return (
    <div className="interactive-builder">
      <h3>Build Your Own Tensor</h3>
      <p className="pipeline-subtitle">
        Define a tensor, add operations, and watch step by step how the engine works
      </p>

      <div className="examples-section">
        <span className="examples-label">Quick start:</span>
        <div className="preset-selector">
          {examples.map((ex, i) => (
            <button key={i} onClick={() => loadExample(ex)}>{ex.name}</button>
          ))}
        </div>
      </div>

      {explanation && <div className="explanation-box">{explanation}</div>}

      <div className="builder-inputs">
        <div className="input-group">
          <label>Shape:</label>
          <input value={shape} onChange={e => setShape(e.target.value)} placeholder="2,3,4" />
        </div>
        <div className="input-group">
          <label>DType:</label>
          <select value={dtype} onChange={e => setDtype(e.target.value)}>
            <option value="Float32">Float32</option>
            <option value="Float16">Float16</option>
            <option value="Int32">Int32</option>
            <option value="Int64">Int64</option>
            <option value="Int8">Int8</option>
            <option value="Bool">Bool</option>
          </select>
        </div>
        <div className="input-group">
          <label>Fill:</label>
          <select value={fill} onChange={e => setFill(e.target.value)}>
            <option value="zeros">zeros</option>
            <option value="ones">ones</option>
          </select>
        </div>
      </div>

      <div className="op-builder">
        <div className="op-add-row">
          <select value={opType} onChange={e => { setOpType(e.target.value); setOpParam(""); }}>
            <option value="access">Access element</option>
            <option value="reshape">Reshape</option>
            <option value="slice">Slice</option>
            <option value="transpose">Transpose</option>
            <option value="broadcast">Broadcast</option>
            <option value="iterate">Iterate</option>
            <option value="contiguous">Make contiguous</option>
            <option value="clone">Clone</option>
          </select>
          {!["iterate", "contiguous", "clone"].includes(opType) && (
            <input
              value={opParam}
              onChange={e => setOpParam(e.target.value)}
              placeholder={opPlaceholders[opType]}
            />
          )}
          <button onClick={addOperation} className="add-btn">+ Add</button>
        </div>

        {ops.length > 0 && (
          <div className="op-queue">
            <span className="queue-label">Operations:</span>
            {ops.map((op, i) => (
              <div key={i} className="queued-op">
                <span>{op.type}{op.param ? `(${op.param})` : ""}</span>
                <button onClick={() => removeOp(i)} className="remove-btn">×</button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="run-section">
        <button onClick={runPipeline} className="run-btn" disabled={!wasmLoaded}>
          {wasmLoaded ? "Run Pipeline" : "Loading WASM..."}
        </button>
      </div>

      {/* Step-by-step visual output */}
      {running && trace.length > 0 && (
        <div className="step-viewer">
          <div className="step-nav">
            <button onClick={prevStep} disabled={currentStep === 0}>← Previous</button>
            <span className="step-indicator">
              Step {currentStep + 1} / {trace.length}
              <span className="step-component-name"> — {trace[currentStep].component}</span>
            </span>
            <button onClick={nextStep} disabled={currentStep === trace.length - 1}>Next →</button>
            <button onClick={showAll} className="show-all-btn">Show All</button>
          </div>

          {/* Progress bar */}
          <div className="step-progress">
            {trace.map((s, i) => (
              <div
                key={i}
                className={`progress-dot ${i <= currentStep ? "done" : ""} ${i === currentStep ? "current" : ""}`}
                title={s.component}
                onClick={() => setCurrentStep(i)}
              />
            ))}
          </div>

          {/* Current step visual */}
          <div className="step-visual-container">
            <div className="step-headline">{trace[currentStep].title}</div>
            <div className="step-subline">{trace[currentStep].detail}</div>
            <StepVisual step={trace[currentStep]} />
          </div>

          {currentStep === trace.length - 1 && (
            <div className="pipeline-done">
              Pipeline complete — {trace.length} steps executed from C++
            </div>
          )}
        </div>
      )}
    </div>
  );
}
