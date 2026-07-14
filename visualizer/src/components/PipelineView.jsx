import { useState, useEffect } from "react";
import {
  initWasm,
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

function StepCard({ step, index, isActive, onClick }) {
  const componentColors = {
    DType: "#d29922",
    Shape: "#58a6ff",
    Stride: "#a371f7",
    TensorStorage: "#3fb950",
    Tensor: "#f0f6fc",
    "Tensor::operator()": "#ff7b72",
    "Tensor::reshape": "#79c0ff",
    "Tensor::slice": "#7ee787",
    "Tensor::transpose": "#d2a8ff",
    "Tensor::contiguous": "#ffa657",
    "Tensor::clone": "#ffa657",
    BroadcastEngine: "#d29922",
    TensorIterator: "#f778ba",
    Error: "#f85149",
  };

  const color = componentColors[step.component] || "#8b949e";

  return (
    <div
      className={`pipeline-step ${isActive ? "active" : ""} ${step.component === "Error" ? "error" : ""}`}
      onClick={onClick}
    >
      <div className="step-header">
        <div className="step-index">{index + 1}</div>
        <div className="step-component" style={{ color }}>{step.component}</div>
        <div className="step-title-text">{step.title}</div>
      </div>
      <div className="step-detail-text">{step.detail}</div>

      {isActive && (
        <div className="step-expanded">
          <div className="step-internal">
            {step.internal.map((line, i) => (
              <div key={i} className={`internal-line ${line === "" ? "blank" : ""}`}>
                {line}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function PipelineConnector() {
  return (
    <div className="pipeline-connector">
      <div className="connector-line"></div>
      <div className="connector-arrow">▼</div>
    </div>
  );
}

const presets = [
  {
    name: "Create Tensor",
    description: "Full creation flow: DType → Shape → Stride → Storage → Tensor",
    run: () => traceFullCreate("Float32", [2, 3, 4], "zeros"),
  },
  {
    name: "Element Access",
    description: "How t(1, 2, 3) computes the memory address",
    run: () => [
      ...traceFullCreate("Float32", [2, 3, 4], "zeros"),
      ...traceAccess([2, 3, 4], [1, 2, 3]),
    ],
  },
  {
    name: "Reshape (zero-copy)",
    description: "Reshape [2,3,4] → [6,4]: only metadata changes",
    run: () => [
      ...traceFullCreate("Float32", [2, 3, 4], "ones"),
      ...traceReshape([2, 3, 4], [6, 4]),
    ],
  },
  {
    name: "Slice",
    description: "Slice dim 0, range [1:3]: pointer offset, no copy",
    run: () => [
      ...traceFullCreate("Float32", [4, 6], "zeros"),
      ...traceSlice([4, 6], 0, 1, 3),
    ],
  },
  {
    name: "Transpose",
    description: "Transpose dims 0↔1: becomes non-contiguous",
    run: () => [
      ...traceFullCreate("Float32", [2, 3, 4], "zeros"),
      ...traceTranspose([2, 3, 4], 0, 1),
    ],
  },
  {
    name: "Broadcast",
    description: "Broadcast [1,3] with [4,1] → virtual strides",
    run: () => traceBroadcast([1, 3], [4, 1]),
  },
  {
    name: "Iterator",
    description: "Contiguous vs strided iteration paths",
    run: () => [
      ...traceIterator([2, 3, 4], false),
      ...traceIterator([2, 3, 4], true),
    ],
  },
  {
    name: "Contiguous Copy",
    description: "Make a transposed tensor contiguous (requires data copy)",
    run: () => [
      ...traceFullCreate("Float32", [3, 4], "ones"),
      ...traceTranspose([3, 4], 0, 1),
      ...traceContiguous([3, 4]),
    ],
  },
  {
    name: "Clone (deep copy)",
    description: "Independent copy with new storage",
    run: () => [
      ...traceFullCreate("Float32", [4, 4], "ones"),
      ...traceClone([4, 4]),
    ],
  },
  {
    name: "Full Pipeline",
    description: "Create → Access → Reshape → Slice → Transpose → Contiguous → Clone",
    run: () => [
      ...traceFullCreate("Float32", [2, 3, 4], "ones"),
      ...traceAccess([2, 3, 4], [1, 0, 2]),
      ...traceReshape([2, 3, 4], [6, 4]),
      ...traceSlice([6, 4], 0, 1, 5),
      ...traceTranspose([4, 4], 0, 1),
      ...traceContiguous([4, 4]),
      ...traceClone([4, 4]),
      ...traceBroadcast([4, 4], [1, 4]),
    ],
  },
];

export default function PipelineView() {
  const [wasmLoaded, setWasmLoaded] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [selectedPreset, setSelectedPreset] = useState(0);
  const [trace, setTrace] = useState([]);
  const [activeStep, setActiveStep] = useState(null);
  const [autoPlay, setAutoPlay] = useState(false);
  const [playIndex, setPlayIndex] = useState(-1);

  useEffect(() => {
    initWasm()
      .then(() => {
        setWasmLoaded(true);
        setTrace(presets[0].run());
      })
      .catch((err) => setLoadError(err.message));
  }, []);

  function selectPreset(i) {
    setSelectedPreset(i);
    setTrace(presets[i].run());
    setPlayIndex(-1);
    setActiveStep(null);
    setAutoPlay(false);
  }

  function handlePlay() {
    setAutoPlay(true);
    setPlayIndex(0);
    const interval = setInterval(() => {
      setPlayIndex((prev) => {
        if (prev >= trace.length - 1) {
          clearInterval(interval);
          setAutoPlay(false);
          return prev;
        }
        return prev + 1;
      });
    }, 600);
  }

  function handleReset() {
    setPlayIndex(-1);
    setActiveStep(null);
    setAutoPlay(false);
  }

  if (loadError) {
    return <div className="error">Failed to load C++ WASM module: {loadError}</div>;
  }

  if (!wasmLoaded) {
    return <div className="loading">Loading C++ tensor engine (WebAssembly)...</div>;
  }

  const visibleSteps = autoPlay || playIndex >= 0 ? playIndex + 1 : trace.length;

  return (
    <div className="pipeline-view">
      <h3>Tensor Engine Pipeline <span className="wasm-badge">C++ WASM</span></h3>
      <p className="pipeline-subtitle">
        Every step runs actual compiled C++ code — Shape, Stride, Storage, Tensor, Broadcast, Iterator
      </p>

      <div className="preset-selector">
        {presets.map((preset, i) => (
          <button
            key={i}
            className={selectedPreset === i ? "active" : ""}
            onClick={() => selectPreset(i)}
          >
            {preset.name}
          </button>
        ))}
      </div>

      <div className="preset-description">{presets[selectedPreset].description}</div>

      <div className="play-controls">
        <button onClick={handlePlay} disabled={autoPlay}>▶ Play Step-by-Step</button>
        <button onClick={() => setPlayIndex(trace.length - 1)}>⏭ Show All</button>
        <button onClick={handleReset}>↺ Reset</button>
        <span className="step-counter">{Math.max(0, visibleSteps)} / {trace.length} steps</span>
      </div>

      <div className="pipeline-flow">
        {trace.slice(0, visibleSteps).map((step, i) => (
          <div key={i}>
            {i > 0 && <PipelineConnector />}
            <StepCard
              step={step}
              index={i}
              isActive={activeStep === i || playIndex === i}
              onClick={() => setActiveStep(activeStep === i ? null : i)}
            />
          </div>
        ))}
      </div>

      {visibleSteps === trace.length && trace.length > 0 && (
        <div className="pipeline-summary">
          <div className="summary-title">Pipeline Complete</div>
          <div className="summary-stats">
            <span>{trace.length} steps from compiled C++</span>
          </div>
        </div>
      )}
    </div>
  );
}
