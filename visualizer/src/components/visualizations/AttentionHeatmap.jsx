import { useState, useEffect, useRef } from "react";
import "./Visualizations.css";

/**
 * Animated attention heatmap visualization.
 * Shows how tokens attend to each other with an animated highlight.
 */
export default function AttentionHeatmap({ tokens = [], size = 4 }) {
  const [activeRow, setActiveRow] = useState(null);
  const [animating, setAnimating] = useState(false);
  const [weights, setWeights] = useState([]);
  const timerRef = useRef(null);

  // Generate deterministic attention weights
  useEffect(() => {
    const n = tokens.length || size;
    const w = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      let sum = 0;
      for (let j = 0; j < n; j++) {
        // Tokens near each other attend more (distance-biased)
        const dist = Math.abs(i - j);
        const raw = Math.exp(-dist * 0.5) + Math.random() * 0.3;
        row.push(raw);
        sum += raw;
      }
      // Normalize to probabilities
      w.push(row.map((v) => v / sum));
    }
    setWeights(w);
  }, [tokens.length, size]);

  const labels = tokens.length > 0 ? tokens : Array.from({ length: size }, (_, i) => `t${i}`);
  const n = labels.length;

  // Animate through rows
  function startAnimation() {
    setAnimating(true);
    let row = 0;
    setActiveRow(0);
    timerRef.current = setInterval(() => {
      row++;
      if (row >= n) {
        clearInterval(timerRef.current);
        setAnimating(false);
        setActiveRow(null);
      } else {
        setActiveRow(row);
      }
    }, 800);
  }

  useEffect(() => {
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, []);

  if (weights.length === 0) return null;

  return (
    <div className="viz-container">
      <div className="viz-header">
        <h4 className="viz-title">Attention Heatmap</h4>
        <button
          className="viz-btn"
          onClick={startAnimation}
          disabled={animating}
        >
          {animating ? "Animating..." : "▶ Animate"}
        </button>
      </div>
      <p className="viz-desc">
        Each cell shows how much token (row) attends to token (column).
        Brighter = stronger attention.
      </p>

      <div className="heatmap-wrapper">
        {/* Column labels */}
        <div className="heatmap-col-labels" style={{ gridTemplateColumns: `60px repeat(${n}, 1fr)` }}>
          <div />
          {labels.map((l, i) => (
            <div key={i} className="heatmap-label col-label">{l}</div>
          ))}
        </div>

        {/* Heatmap grid */}
        <div className="heatmap-grid" style={{ gridTemplateColumns: `60px repeat(${n}, 1fr)` }}>
          {weights.map((row, i) => (
            <div key={i} className="heatmap-row" style={{ display: "contents" }}>
              <div className={`heatmap-label row-label ${activeRow === i ? "active" : ""}`}>
                {labels[i]}
              </div>
              {row.map((val, j) => (
                <div
                  key={j}
                  className={`heatmap-cell ${activeRow === i ? "row-active" : ""} ${
                    activeRow === i && val > 0.2 ? "highlight" : ""
                  }`}
                  style={{
                    "--intensity": val,
                    backgroundColor: `rgba(74, 144, 226, ${val})`,
                  }}
                  title={`${labels[i]} → ${labels[j]}: ${(val * 100).toFixed(1)}%`}
                >
                  <span className="cell-value">{(val * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {activeRow !== null && (
        <div className="viz-annotation">
          Token "<strong>{labels[activeRow]}</strong>" is attending to all other tokens.
          Highest attention: "{labels[weights[activeRow].indexOf(Math.max(...weights[activeRow]))]}"
        </div>
      )}
    </div>
  );
}
