import { useState, useMemo } from "react";
import "./QuantizeViz.css";

// Generate fake "weights" that look like a trained neural network
function generateWeights(n, seed = 42) {
  const weights = [];
  let s = seed;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    const u = (s >>> 16) / 65536;
    // Box-Muller for normal distribution
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    const v = (s >>> 16) / 65536;
    weights.push(Math.sqrt(-2 * Math.log(u + 0.001)) * Math.cos(2 * Math.PI * v) * 0.1);
  }
  return weights;
}

// Symmetric quantization
function quantizeSymmetric(weights, bits) {
  const maxVal = Math.max(...weights.map(Math.abs));
  const levels = Math.pow(2, bits - 1) - 1; // 127 for INT8
  const scale = maxVal / levels;
  const quantized = weights.map(w => {
    const q = Math.round(w / scale);
    return Math.max(-levels - 1, Math.min(levels, q));
  });
  const dequantized = quantized.map(q => q * scale);
  return { quantized, dequantized, scale, levels };
}

// Compute histogram
function histogram(values, bins = 50) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const idx = Math.min(bins - 1, Math.floor(((v - min) / range) * bins));
    counts[idx]++;
  }
  return { counts, min, max };
}

// Compute error metrics
function computeError(original, dequantized) {
  let sumSqErr = 0, sumSqSig = 0, maxErr = 0;
  for (let i = 0; i < original.length; i++) {
    const err = Math.abs(original[i] - dequantized[i]);
    sumSqErr += err * err;
    sumSqSig += original[i] * original[i];
    maxErr = Math.max(maxErr, err);
  }
  const rmse = Math.sqrt(sumSqErr / original.length);
  const snr = sumSqErr > 0 ? 10 * Math.log10(sumSqSig / sumSqErr) : Infinity;
  return { rmse, snr, maxErr };
}

export default function QuantizeViz() {
  const [bits, setBits] = useState(8);
  const [numWeights] = useState(2000);

  const weights = useMemo(() => generateWeights(numWeights), [numWeights]);

  const result = useMemo(() => {
    const { quantized, dequantized, scale, levels } = quantizeSymmetric(weights, bits);
    const error = computeError(weights, dequantized);
    const origHist = histogram(weights);
    const quantHist = histogram(dequantized);
    return { quantized, dequantized, scale, levels, error, origHist, quantHist };
  }, [weights, bits]);

  const compressionRatio = 32 / bits;
  const memoryOriginal = numWeights * 4;
  const memoryQuantized = numWeights * (bits / 8);

  return (
    <div className="qviz">
      <div className="qviz-header">
        <h1><span className="qviz-icon">📦</span> INT8 Quantization Explorer</h1>
        <p>See how neural network weights compress from 32-bit to 8-bit with minimal accuracy loss</p>
      </div>

      {/* Bit Width Selector */}
      <div className="qviz-controls">
        <div className="bits-selector">
          <label>Precision:</label>
          {[2, 4, 8, 16, 32].map(b => (
            <button
              key={b}
              className={`bits-btn ${bits === b ? "active" : ""}`}
              onClick={() => setBits(b)}
            >
              {b}-bit
            </button>
          ))}
        </div>
        <div className="control-info">
          {bits === 32 ? "Full precision (baseline)" : `Compressed ${compressionRatio}×`}
        </div>
      </div>

      {/* Histograms */}
      <div className="qviz-histograms">
        <div className="hist-panel">
          <div className="hist-title">Original Weights (Float32)</div>
          <div className="hist-chart">
            {result.origHist.counts.map((c, i) => (
              <div
                key={i}
                className="hist-bar orig"
                style={{ height: `${(c / Math.max(...result.origHist.counts)) * 100}%` }}
              />
            ))}
          </div>
          <div className="hist-labels">
            <span>{result.origHist.min.toFixed(3)}</span>
            <span>0</span>
            <span>{result.origHist.max.toFixed(3)}</span>
          </div>
        </div>

        <div className="hist-arrow">→</div>

        <div className="hist-panel">
          <div className="hist-title">Dequantized ({bits}-bit)</div>
          <div className="hist-chart">
            {result.quantHist.counts.map((c, i) => (
              <div
                key={i}
                className="hist-bar quant"
                style={{ height: `${(c / Math.max(...result.quantHist.counts)) * 100}%` }}
              />
            ))}
          </div>
          <div className="hist-labels">
            <span>{result.quantHist.min.toFixed(3)}</span>
            <span>0</span>
            <span>{result.quantHist.max.toFixed(3)}</span>
          </div>
        </div>
      </div>

      {/* Metrics Dashboard */}
      <div className="qviz-metrics">
        <div className="qm-card">
          <div className="qm-label">Compression</div>
          <div className="qm-value">{compressionRatio}×</div>
          <div className="qm-sub">{memoryOriginal} B → {memoryQuantized} B</div>
        </div>
        <div className="qm-card">
          <div className="qm-label">SNR</div>
          <div className={`qm-value ${result.error.snr > 30 ? "good" : result.error.snr > 15 ? "ok" : "bad"}`}>
            {result.error.snr === Infinity ? "∞" : result.error.snr.toFixed(1)} dB
          </div>
          <div className="qm-sub">{result.error.snr > 30 ? "Excellent" : result.error.snr > 15 ? "Good" : "Lossy"}</div>
        </div>
        <div className="qm-card">
          <div className="qm-label">Max Error</div>
          <div className="qm-value">{result.error.maxErr.toFixed(5)}</div>
          <div className="qm-sub">worst-case deviation</div>
        </div>
        <div className="qm-card">
          <div className="qm-label">Scale Factor</div>
          <div className="qm-value">{result.scale.toExponential(2)}</div>
          <div className="qm-sub">float = int × scale</div>
        </div>
      </div>

      {/* How it works */}
      <div className="qviz-how">
        <h3>How Symmetric Quantization Works</h3>
        <div className="how-steps">
          <div className="how-step">
            <div className="step-num">1</div>
            <div className="step-content">
              <strong>Find scale</strong>
              <code>scale = max(|weights|) / 127</code>
              <span>Maps the range [-max, +max] to [-127, +127]</span>
            </div>
          </div>
          <div className="how-step">
            <div className="step-num">2</div>
            <div className="step-content">
              <strong>Quantize</strong>
              <code>q[i] = clamp(round(w[i] / scale), -128, 127)</code>
              <span>Float → Int8 (4 bytes → 1 byte)</span>
            </div>
          </div>
          <div className="how-step">
            <div className="step-num">3</div>
            <div className="step-content">
              <strong>Compute in INT8</strong>
              <code>C_int32 = A_int8 × B_int8</code>
              <span>Integer math is 2-4× faster than float</span>
            </div>
          </div>
          <div className="how-step">
            <div className="step-num">4</div>
            <div className="step-content">
              <strong>Dequantize</strong>
              <code>result[i] = C_int32[i] × scale_a × scale_b</code>
              <span>Back to float for the next layer</span>
            </div>
          </div>
        </div>
      </div>

      {/* Real-world impact */}
      <div className="qviz-impact">
        <h3>Real-World Memory Savings</h3>
        <div className="impact-grid">
          <div className="impact-item">
            <div className="impact-model">GPT-2 Small</div>
            <div className="impact-bar-wrap">
              <div className="impact-bar fp32" style={{ width: "100%" }}>500 MB</div>
            </div>
            <div className="impact-bar-wrap">
              <div className="impact-bar int8" style={{ width: "25%" }}>125 MB</div>
            </div>
          </div>
          <div className="impact-item">
            <div className="impact-model">LLaMA-7B</div>
            <div className="impact-bar-wrap">
              <div className="impact-bar fp32" style={{ width: "100%" }}>14 GB</div>
            </div>
            <div className="impact-bar-wrap">
              <div className="impact-bar int8" style={{ width: "25%" }}>3.5 GB</div>
            </div>
          </div>
          <div className="impact-item">
            <div className="impact-model">ResNet-50</div>
            <div className="impact-bar-wrap">
              <div className="impact-bar fp32" style={{ width: "100%" }}>100 MB</div>
            </div>
            <div className="impact-bar-wrap">
              <div className="impact-bar int8" style={{ width: "25%" }}>25 MB</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
