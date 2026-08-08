import { useState, useEffect, useRef } from "react";
import "./Visualizations.css";

/**
 * Animated matrix multiplication visualization.
 * Shows row × column dot products with highlighted cells.
 */
export default function MatMulAnimation({ M = 3, K = 3, N = 3 }) {
  const [A, setA] = useState([]);
  const [B, setB] = useState([]);
  const [C, setC] = useState([]);
  const [activeI, setActiveI] = useState(-1);
  const [activeJ, setActiveJ] = useState(-1);
  const [animating, setAnimating] = useState(false);
  const [dotProducts, setDotProducts] = useState([]);
  const timerRef = useRef(null);

  // Generate random matrices on mount
  useEffect(() => {
    const genMat = (rows, cols) =>
      Array.from({ length: rows * cols }, () =>
        parseFloat((Math.random() * 2 - 1).toFixed(2))
      );
    setA(genMat(M, K));
    setB(genMat(K, N));
    setC(new Array(M * N).fill(null));
    setDotProducts([]);
  }, [M, K, N]);

  function startAnimation() {
    setAnimating(true);
    setC(new Array(M * N).fill(null));
    setDotProducts([]);
    let idx = 0;
    const total = M * N;

    function step() {
      if (idx >= total) {
        setActiveI(-1);
        setActiveJ(-1);
        setAnimating(false);
        return;
      }
      const i = Math.floor(idx / N);
      const j = idx % N;
      setActiveI(i);
      setActiveJ(j);

      // Compute dot product
      let sum = 0;
      const products = [];
      for (let k = 0; k < K; k++) {
        const prod = A[i * K + k] * B[k * N + j];
        products.push({ a: A[i * K + k], b: B[k * N + j], prod });
        sum += prod;
      }

      setDotProducts(products);
      setC((prev) => {
        const next = [...prev];
        next[i * N + j] = parseFloat(sum.toFixed(3));
        return next;
      });

      idx++;
      timerRef.current = setTimeout(step, 600);
    }

    step();
  }

  useEffect(() => {
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, []);

  return (
    <div className="viz-container">
      <div className="viz-header">
        <h4 className="viz-title">Matrix Multiplication</h4>
        <button className="viz-btn" onClick={startAnimation} disabled={animating}>
          {animating ? "Computing..." : "▶ Animate"}
        </button>
      </div>
      <p className="viz-desc">
        Watch each dot product computed one by one. Row from A × Column from B = one element of C.
      </p>

      <div className="matmul-layout">
        {/* Matrix A */}
        <div className="mat-panel">
          <span className="mat-label">A [{M}×{K}]</span>
          <div className="mat-grid" style={{ gridTemplateColumns: `repeat(${K}, 1fr)` }}>
            {A.map((v, idx) => {
              const row = Math.floor(idx / K);
              const isRowActive = row === activeI;
              return (
                <div
                  key={idx}
                  className={`mat-cell ${isRowActive ? "row-highlight" : ""}`}
                >
                  {v}
                </div>
              );
            })}
          </div>
        </div>

        <span className="mat-operator">×</span>

        {/* Matrix B */}
        <div className="mat-panel">
          <span className="mat-label">B [{K}×{N}]</span>
          <div className="mat-grid" style={{ gridTemplateColumns: `repeat(${N}, 1fr)` }}>
            {B.map((v, idx) => {
              const col = idx % N;
              const isColActive = col === activeJ;
              return (
                <div
                  key={idx}
                  className={`mat-cell ${isColActive ? "col-highlight" : ""}`}
                >
                  {v}
                </div>
              );
            })}
          </div>
        </div>

        <span className="mat-operator">=</span>

        {/* Matrix C (result) */}
        <div className="mat-panel">
          <span className="mat-label">C [{M}×{N}]</span>
          <div className="mat-grid" style={{ gridTemplateColumns: `repeat(${N}, 1fr)` }}>
            {C.map((v, idx) => {
              const row = Math.floor(idx / N);
              const col = idx % N;
              const isActive = row === activeI && col === activeJ;
              return (
                <div
                  key={idx}
                  className={`mat-cell ${isActive ? "computing" : ""} ${v !== null ? "filled" : "empty"}`}
                >
                  {v !== null ? v : "·"}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Dot product detail */}
      {dotProducts.length > 0 && (
        <div className="dot-product-detail">
          <span className="dp-label">
            C[{activeI}][{activeJ}] =
          </span>
          <div className="dp-terms">
            {dotProducts.map((d, i) => (
              <span key={i} className="dp-term">
                {i > 0 && <span className="dp-op">+</span>}
                <span className="dp-a">{d.a}</span>
                <span className="dp-times">×</span>
                <span className="dp-b">{d.b}</span>
              </span>
            ))}
            <span className="dp-result">
              = {dotProducts.reduce((s, d) => s + d.prod, 0).toFixed(3)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
