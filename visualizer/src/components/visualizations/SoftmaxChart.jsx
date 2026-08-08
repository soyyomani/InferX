import { useState, useEffect, useRef } from "react";
import "./Visualizations.css";

/**
 * Animated softmax visualization.
 * Shows logits → exp → normalize → probabilities with animated bars.
 */
export default function SoftmaxChart({ logits: initialLogits }) {
  const [logits, setLogits] = useState(initialLogits || [2.0, 1.0, 0.1, -1.0, 3.0]);
  const [stage, setStage] = useState(0); // 0=logits, 1=shifted, 2=exp, 3=probs
  const [animating, setAnimating] = useState(false);
  const timerRef = useRef(null);

  const maxLogit = Math.max(...logits);
  const shifted = logits.map((v) => v - maxLogit);
  const exps = shifted.map((v) => Math.exp(v));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / sumExp);

  const stageLabels = [
    "Raw Logits (Neural Network Output)",
    "After Subtracting Max (Stability)",
    "After Exponentiation (e^x)",
    "Final Probabilities (Normalized)",
  ];

  function getValues() {
    switch (stage) {
      case 0: return logits;
      case 1: return shifted;
      case 2: return exps;
      case 3: return probs;
      default: return logits;
    }
  }

  function getMaxForScale() {
    const vals = getValues();
    return Math.max(...vals.map(Math.abs), 0.001);
  }

  function startAnimation() {
    setAnimating(true);
    setStage(0);
    let s = 0;
    timerRef.current = setInterval(() => {
      s++;
      if (s > 3) {
        clearInterval(timerRef.current);
        setAnimating(false);
      } else {
        setStage(s);
      }
    }, 1200);
  }

  useEffect(() => {
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, []);

  const values = getValues();
  const maxVal = getMaxForScale();
  const argmax = probs.indexOf(Math.max(...probs));

  return (
    <div className="viz-container">
      <div className="viz-header">
        <h4 className="viz-title">Softmax Transformation</h4>
        <button className="viz-btn" onClick={startAnimation} disabled={animating}>
          {animating ? "Transforming..." : "▶ Animate Steps"}
        </button>
      </div>
      <p className="viz-desc">{stageLabels[stage]}</p>

      {/* Stage indicators */}
      <div className="softmax-stages">
        {stageLabels.map((label, i) => (
          <button
            key={i}
            className={`stage-pill ${stage === i ? "active" : ""} ${i < stage ? "done" : ""}`}
            onClick={() => { setStage(i); setAnimating(false); if (timerRef.current) clearInterval(timerRef.current); }}
          >
            {i + 1}. {["Logits", "Shift", "Exp", "Probs"][i]}
          </button>
        ))}
      </div>

      {/* Bar Chart */}
      <div className="softmax-bars">
        {values.map((v, i) => {
          const barWidth = stage === 3
            ? v * 100 // probabilities are 0-1
            : (Math.abs(v) / maxVal) * 80;
          const isNeg = v < 0;
          const isWinner = stage === 3 && i === argmax;

          return (
            <div key={i} className={`bar-row ${isWinner ? "winner" : ""}`}>
              <span className="bar-index">{i}</span>
              <div className="bar-track">
                <div
                  className={`bar-fill-animated ${isNeg ? "negative" : "positive"} ${isWinner ? "glow" : ""}`}
                  style={{
                    width: `${barWidth}%`,
                    transitionDelay: `${i * 50}ms`,
                  }}
                />
              </div>
              <span className="bar-value">
                {stage === 3
                  ? `${(v * 100).toFixed(1)}%`
                  : v.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>

      {/* Formula annotation */}
      <div className="softmax-formula">
        {stage === 0 && <span>These raw scores can be any value (-∞ to +∞)</span>}
        {stage === 1 && <span>Subtract max ({maxLogit.toFixed(2)}) for numerical stability</span>}
        {stage === 2 && <span>e^x makes all values positive. Sum = {sumExp.toFixed(3)}</span>}
        {stage === 3 && (
          <span>
            Divide by sum → valid probabilities. Winner: index {argmax} ({(probs[argmax] * 100).toFixed(1)}%)
          </span>
        )}
      </div>
    </div>
  );
}
