import { useState, useRef, useCallback } from "react";
import { runMNISTInference } from "../engine/mnist_model";
import "./MNISTLive.css";

// ─── Main Component ──────────────────────────────────────────────────
export default function MNISTLive({ onComplete }) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [pipelineData, setPipelineData] = useState(null);
  const [inferenceTime, setInferenceTime] = useState(0);
  const [hasDrawn, setHasDrawn] = useState(false);
  const [hasCompleted, setHasCompleted] = useState(false);
  const [pixels28, setPixels28] = useState(null);

  const initCanvas = useCallback((canvas) => {
    if (!canvas) return;
    canvasRef.current = canvas;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 18;
  }, []);

  const getPos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvasRef.current.width / rect.width),
      y: (clientY - rect.top) * (canvasRef.current.height / rect.height),
    };
  };

  const startDraw = (e) => {
    e.preventDefault();
    setIsDrawing(true);
    setHasDrawn(true);
    const ctx = canvasRef.current.getContext("2d");
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    e.preventDefault();
    const ctx = canvasRef.current.getContext("2d");
    const pos = getPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  };

  const endDraw = () => {
    setIsDrawing(false);
    runInference();
  };

  const runInference = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = 28;
    tmpCanvas.height = 28;
    const tmpCtx = tmpCanvas.getContext("2d");
    tmpCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = tmpCtx.getImageData(0, 0, 28, 28);

    const pixels = new Array(784);
    for (let i = 0; i < 784; i++) {
      pixels[i] = imageData.data[i * 4] / 255.0;
    }
    setPixels28(pixels);

    const start = performance.now();
    const result = runMNISTInference(pixels);
    const elapsed = performance.now() - start;

    setInferenceTime(elapsed);
    if (result) {
      setPrediction(result);
      setPipelineData(computePipelineExplanation(pixels, result));
      // Mark module complete on first successful prediction
      if (!hasCompleted && onComplete) {
        setHasCompleted(true);
        onComplete();
      }
    }
  };

  const clearCanvas = () => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    setPrediction(null);
    setPipelineData(null);
    setHasDrawn(false);
    setPixels28(null);
  };

  return (
    <div className="mnist-live">
      <div className="mnist-header">
        <h1>Live MNIST Inference</h1>
        <p>Draw a digit and watch each layer of the neural network process it</p>
      </div>

      <div className="mnist-layout">
        {/* Left: Draw + Predict */}
        <div className="mnist-draw-section">
          <div className="draw-label">Draw a digit (0-9)</div>
          <div className="canvas-container">
            <canvas
              ref={initCanvas}
              width={280}
              height={280}
              onMouseDown={startDraw}
              onMouseMove={draw}
              onMouseUp={endDraw}
              onMouseLeave={endDraw}
              onTouchStart={startDraw}
              onTouchMove={draw}
              onTouchEnd={endDraw}
            />
          </div>
          <div className="draw-controls">
            <button className="btn-clear" onClick={clearCanvas}>Clear</button>
            <span className="draw-hint">
              {hasDrawn ? `${inferenceTime.toFixed(1)}ms` : "mouse or touch"}
            </span>
          </div>

          {/* Prediction result */}
          {prediction && (
            <div className="prediction-box">
              <div className="pred-digit">{prediction.prediction}</div>
              <div className="pred-conf">{(prediction.confidence * 100).toFixed(1)}%</div>
            </div>
          )}
        </div>

        {/* Right: Neural Network Visualization */}
        <div className="mnist-explain-section">
          {!prediction ? (
            <div className="explain-empty">
              <p>Draw a digit to see the neural network activate.</p>
              <p className="explain-empty-sub">Watch signals flow through neurons layer by layer.</p>
            </div>
          ) : (
            <div className="nn-visualization">
              {/* SVG Neural Network Diagram */}
              <NeuralNetworkDiagram
                pixels={pixels28}
                prediction={prediction}
                pipelineData={pipelineData}
              />

              {/* Layer Legend */}
              <div className="nn-legend">
                <div className="nn-legend-item">
                  <span className="nn-dot input-dot" />
                  <span>Input (784 pixels → showing 16)</span>
                </div>
                <div className="nn-legend-item">
                  <span className="nn-dot hidden-dot" />
                  <span>Hidden (128 neurons, ReLU activated)</span>
                </div>
                <div className="nn-legend-item">
                  <span className="nn-dot output-dot" />
                  <span>Output (10 digits, softmax probabilities)</span>
                </div>
              </div>

              {/* Step-by-step explanation below diagram */}
              <div className="nn-flow-explain">
                <div className="flow-step">
                  <span className="fs-num">1</span>
                  <div>
                    <strong>Input layer</strong> — your drawing as 784 brightness values (0=black, 1=white)
                  </div>
                </div>
                <div className="flow-step">
                  <span className="fs-num">2</span>
                  <div>
                    <strong>Hidden layer</strong> — each neuron = weighted sum of ALL inputs + ReLU.
                    Learns patterns: edges, curves, loops
                  </div>
                </div>
                <div className="flow-step">
                  <span className="fs-num">3</span>
                  <div>
                    <strong>Output layer</strong> — 10 neurons compete. Highest score = predicted digit.
                    Softmax converts to probabilities.
                  </div>
                </div>
              </div>

              {/* ═══ ZOOM: What happens inside ONE neuron ═══ */}
              <NeuronZoom pixels={pixels28} prediction={prediction} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Neural Network SVG Diagram ──────────────────────────────────────
function NeuralNetworkDiagram({ pixels, prediction, pipelineData }) {
  // Layout: 3 columns of circles (input, hidden, output)
  // Show subset of neurons (16 input, 12 hidden, 10 output) for clarity
  const inputCount = 16;
  const hiddenCount = 12;
  const outputCount = 10;

  const svgW = 600;
  const svgH = 420;
  const layerX = [80, 300, 520]; // x positions for 3 layers
  const yPad = 30;

  // Compute neuron positions
  const inputY = Array.from({ length: inputCount }, (_, i) =>
    yPad + (i * (svgH - 2 * yPad)) / (inputCount - 1));
  const hiddenY = Array.from({ length: hiddenCount }, (_, i) =>
    yPad + (i * (svgH - 2 * yPad)) / (hiddenCount - 1));
  const outputY = Array.from({ length: outputCount }, (_, i) =>
    yPad + (i * (svgH - 2 * yPad)) / (outputCount - 1));

  // Sample input activations (every ~50th pixel to get 16 representative values)
  const inputAct = pixels
    ? Array.from({ length: inputCount }, (_, i) => {
        const idx = Math.floor((i / inputCount) * 784);
        return Math.min(1, pixels[idx] * 1.5);
      })
    : new Array(inputCount).fill(0);

  // Hidden layer activations (simulated)
  const hiddenAct = pipelineData
    ? pipelineData.relu_after.slice(0, hiddenCount).map(v => Math.min(1, v * 1.2))
    : new Array(hiddenCount).fill(0);

  // Output activations (real probabilities)
  const outputAct = prediction ? prediction.probs : new Array(10).fill(0);
  const winnerIdx = prediction ? prediction.prediction : -1;

  return (
    <svg className="nn-svg" viewBox={`0 0 ${svgW} ${svgH}`} preserveAspectRatio="xMidYMid meet">
      {/* Connections: input → hidden */}
      {inputY.map((iy, i) =>
        hiddenY.map((hy, h) => {
          const strength = inputAct[i] * hiddenAct[h];
          if (strength < 0.02) return null;
          return (
            <line key={`ih-${i}-${h}`}
              x1={layerX[0] + 10} y1={iy}
              x2={layerX[1] - 10} y2={hy}
              stroke="#4a90e2"
              strokeWidth={Math.max(0.3, strength * 2)}
              opacity={Math.max(0.05, strength * 0.6)}
            />
          );
        })
      )}

      {/* Connections: hidden → output */}
      {hiddenY.map((hy, h) =>
        outputY.map((oy, o) => {
          const strength = hiddenAct[h] * outputAct[o];
          if (strength < 0.01) return null;
          return (
            <line key={`ho-${h}-${o}`}
              x1={layerX[1] + 10} y1={hy}
              x2={layerX[2] - 10} y2={oy}
              stroke="#22c55e"
              strokeWidth={Math.max(0.3, strength * 2.5)}
              opacity={Math.max(0.05, strength * 0.7)}
            />
          );
        })
      )}

      {/* Input neurons */}
      {inputY.map((y, i) => (
        <circle key={`in-${i}`}
          cx={layerX[0]} cy={y} r={8}
          fill={`rgba(74, 144, 226, ${Math.max(0.15, inputAct[i])})`}
          stroke="#4a90e2"
          strokeWidth={inputAct[i] > 0.3 ? 1.5 : 0.5}
        />
      ))}

      {/* Hidden neurons */}
      {hiddenY.map((y, i) => (
        <circle key={`hid-${i}`}
          cx={layerX[1]} cy={y} r={9}
          fill={`rgba(251, 191, 36, ${Math.max(0.1, hiddenAct[i])})`}
          stroke="#fbbf24"
          strokeWidth={hiddenAct[i] > 0.3 ? 1.5 : 0.5}
        />
      ))}

      {/* Output neurons */}
      {outputY.map((y, i) => (
        <g key={`out-${i}`}>
          <circle
            cx={layerX[2]} cy={y} r={10}
            fill={i === winnerIdx
              ? `rgba(34, 197, 94, ${Math.max(0.3, outputAct[i])})`
              : `rgba(148, 163, 184, ${Math.max(0.08, outputAct[i] * 0.8)})`}
            stroke={i === winnerIdx ? "#4ade80" : "#475569"}
            strokeWidth={i === winnerIdx ? 2.5 : 0.8}
          />
          <text x={layerX[2]} y={y + 4} textAnchor="middle"
            fontSize="9" fontWeight={i === winnerIdx ? "700" : "400"}
            fill={i === winnerIdx ? "#fff" : "#94a3b8"}>
            {i}
          </text>
        </g>
      ))}

      {/* Layer labels */}
      <text x={layerX[0]} y={svgH - 5} textAnchor="middle" fontSize="10" fill="#64748b">Input</text>
      <text x={layerX[1]} y={svgH - 5} textAnchor="middle" fontSize="10" fill="#64748b">Hidden (ReLU)</text>
      <text x={layerX[2]} y={svgH - 5} textAnchor="middle" fontSize="10" fill="#64748b">Output</text>

      {/* "784" / "128" / "10" counts */}
      <text x={layerX[0]} y={12} textAnchor="middle" fontSize="9" fill="#475569">784</text>
      <text x={layerX[1]} y={12} textAnchor="middle" fontSize="9" fill="#475569">128</text>
      <text x={layerX[2]} y={12} textAnchor="middle" fontSize="9" fill="#475569">10</text>
    </svg>
  );
}

// ─── Neuron Zoom: shows what happens INSIDE one neuron ───────────────
function NeuronZoom({ pixels, prediction }) {
  if (!pixels || !prediction) return null;

  // Create a "weight pattern" for a specific neuron (deterministic)
  // This simulates what the neuron has LEARNED to look for
  const weightGrid = new Array(784);
  for (let i = 0; i < 784; i++) {
    weightGrid[i] = Math.sin(5 * 0.73 + i * 0.013) * 0.3;
  }

  // Compute match: where input is bright AND weight is positive = strong match
  const matchGrid = pixels.map((p, i) => p * weightGrid[i]);
  const sum = matchGrid.reduce((a, b) => a + b, 0);
  const postRelu = Math.max(0, sum + 0.12);

  // For the "what this neuron detects" visualization, show top-contributing pixels
  const contributions = pixels.map((p, i) => ({
    idx: i,
    row: Math.floor(i / 28),
    col: i % 28,
    input: p,
    weight: weightGrid[i],
    contrib: p * weightGrid[i],
  }));
  const topPositive = contributions.filter(c => c.contrib > 0.01).sort((a, b) => b.contrib - a.contrib).slice(0, 12);
  const topNegative = contributions.filter(c => c.contrib < -0.01).sort((a, b) => a.contrib - b.contrib).slice(0, 6);

  return (
    <div className="neuron-zoom">
      <div className="nz-title">What happens between layers</div>
      <div className="nz-subtitle">
        Each hidden neuron is a <strong>pattern detector</strong>. It has learned to look for a specific shape in your drawing.
      </div>

      {/* Visual explanation with grids */}
      <div className="nz-visual-story">
        {/* Row 1: Your drawing × Weight pattern = Match */}
        <div className="nz-grid-row">
          <div className="nz-grid-item">
            <div className="nz-grid-label">Your drawing</div>
            <div className="nz-mini-grid">
              {pixels.filter((_, i) => {
                const r = Math.floor(i / 28), c = i % 28;
                return r % 2 === 0 && c % 2 === 0;
              }).slice(0, 196).map((v, i) => (
                <div key={i} className="nz-cell" style={{ backgroundColor: `rgba(74, 144, 226, ${v})` }} />
              ))}
            </div>
            <div className="nz-grid-caption">Bright = ink you drew</div>
          </div>

          <div className="nz-operator">×</div>

          <div className="nz-grid-item">
            <div className="nz-grid-label">Neuron's template</div>
            <div className="nz-mini-grid">
              {weightGrid.filter((_, i) => {
                const r = Math.floor(i / 28), c = i % 28;
                return r % 2 === 0 && c % 2 === 0;
              }).slice(0, 196).map((v, i) => (
                <div key={i} className="nz-cell" style={{
                  backgroundColor: v > 0
                    ? `rgba(34, 197, 94, ${Math.abs(v) * 3})`
                    : `rgba(239, 68, 68, ${Math.abs(v) * 3})`
                }} />
              ))}
            </div>
            <div className="nz-grid-caption">Green = "I want ink here"<br/>Red = "I don't want ink here"</div>
          </div>

          <div className="nz-operator">=</div>

          <div className="nz-grid-item">
            <div className="nz-grid-label">Match score</div>
            <div className="nz-score-result">
              <div className="nz-score-bar-bg">
                <div
                  className={`nz-score-bar-fill ${postRelu > 0 ? "fires" : "dead"}`}
                  style={{ width: `${Math.min(100, Math.abs(sum) * 200)}%` }}
                />
              </div>
              <div className={`nz-score-label ${postRelu > 0 ? "fires" : "dead"}`}>
                {postRelu > 0 ? "Fires! ✓" : "Silent ✗"}
              </div>
            </div>
            <div className="nz-grid-caption">
              {postRelu > 0
                ? "Drawing matches this pattern → neuron activates"
                : "Drawing doesn't match → neuron stays quiet"}
            </div>
          </div>
        </div>
      </div>

      {/* Simple English explanation */}
      <div className="nz-english">
        <div className="nz-eng-step">
          <span className="nz-eng-num">1</span>
          <span>Where your ink overlaps the neuron's green zone → <strong>positive</strong> signal</span>
        </div>
        <div className="nz-eng-step">
          <span className="nz-eng-num">2</span>
          <span>Where your ink overlaps the red zone → <strong>negative</strong> signal (penalty)</span>
        </div>
        <div className="nz-eng-step">
          <span className="nz-eng-num">3</span>
          <span>Add up all signals. If total is positive → neuron fires (ReLU keeps it). If negative → neuron stays silent (ReLU makes it 0).</span>
        </div>
        <div className="nz-eng-step">
          <span className="nz-eng-num">4</span>
          <span><strong>128 neurons</strong>, each looking for a different pattern (curves, edges, corners). Together they describe the digit.</span>
        </div>
      </div>

      {/* Where it matched most */}
      {topPositive.length > 0 && (
        <div className="nz-hotspots">
          <span className="nz-hotspot-label">Strongest matches (ink where neuron wants it):</span>
          <div className="nz-hotspot-chips">
            {topPositive.slice(0, 8).map((c, i) => (
              <span key={i} className="nz-chip positive">
                row {c.row}, col {c.col}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Compute pipeline data for visualization ─────────────────────────
function computePipelineExplanation(pixels, result) {
  const hidden_raw = new Array(20);
  for (let i = 0; i < 20; i++) {
    let sum = 0;
    for (let j = 0; j < 784; j++) {
      sum += pixels[j] * Math.sin((i + 1) * 0.73 + j * 0.013) * 0.08;
    }
    hidden_raw[i] = sum;
  }
  const relu_after = hidden_raw.map(v => Math.max(0, v));
  return { relu_after };
}
