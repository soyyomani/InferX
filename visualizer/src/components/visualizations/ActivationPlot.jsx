import { useMemo } from "react";
import "./Visualizations.css";

/**
 * SVG-based activation function plot.
 * Shows ReLU, GELU, Sigmoid curves with annotations.
 */
export default function ActivationPlot({ type = "relu" }) {
  const width = 400;
  const height = 220;
  const padding = 40;
  const plotW = width - padding * 2;
  const plotH = height - padding * 2;

  // X range: -3 to 3
  const xMin = -3, xMax = 3;
  const yMin = -1, yMax = 3;

  function xToSvg(x) { return padding + ((x - xMin) / (xMax - xMin)) * plotW; }
  function yToSvg(y) { return padding + plotH - ((y - yMin) / (yMax - yMin)) * plotH; }

  const functions = useMemo(() => {
    const relu = (x) => Math.max(0, x);
    const gelu = (x) => {
      const c = Math.sqrt(2 / Math.PI);
      return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
    };
    const sigmoid = (x) => 1 / (1 + Math.exp(-x));
    const linear = (x) => x;

    return { relu, gelu, sigmoid, linear };
  }, []);

  const fn = functions[type] || functions.relu;

  // Generate curve points
  const points = useMemo(() => {
    const pts = [];
    for (let i = 0; i <= 100; i++) {
      const x = xMin + (i / 100) * (xMax - xMin);
      const y = fn(x);
      pts.push({ x, y, svgX: xToSvg(x), svgY: yToSvg(y) });
    }
    return pts;
  }, [type]);

  const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.svgX} ${p.svgY}`).join(" ");

  // Annotation points
  const annotations = useMemo(() => {
    if (type === "relu") {
      return [
        { x: -2, label: "Dead (output=0)", side: "below" },
        { x: 1.5, label: "Linear (output=x)", side: "above" },
        { x: 0, label: "Kink at 0", side: "above" },
      ];
    }
    if (type === "gelu") {
      return [
        { x: -1.5, label: "Smooth near-zero", side: "below" },
        { x: 1.5, label: "≈ linear for large x", side: "above" },
        { x: 0, label: "GELU(0) = 0", side: "above" },
      ];
    }
    if (type === "sigmoid") {
      return [
        { x: -2, label: "Saturates → 0", side: "below" },
        { x: 2, label: "Saturates → 1", side: "above" },
        { x: 0, label: "σ(0) = 0.5", side: "above" },
      ];
    }
    return [];
  }, [type]);

  const titles = {
    relu: "ReLU: f(x) = max(0, x)",
    gelu: "GELU: f(x) = x × Φ(x)",
    sigmoid: "Sigmoid: f(x) = 1/(1+e^(-x))",
  };

  return (
    <div className="viz-container">
      <div className="viz-header">
        <h4 className="viz-title">{titles[type] || "Activation Function"}</h4>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="activation-svg"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid lines */}
        <line x1={xToSvg(xMin)} y1={yToSvg(0)} x2={xToSvg(xMax)} y2={yToSvg(0)}
          className="axis-line" />
        <line x1={xToSvg(0)} y1={yToSvg(yMin)} x2={xToSvg(0)} y2={yToSvg(yMax)}
          className="axis-line" />

        {/* Grid */}
        {[-2, -1, 1, 2].map((v) => (
          <line key={`gx${v}`}
            x1={xToSvg(v)} y1={yToSvg(yMin)} x2={xToSvg(v)} y2={yToSvg(yMax)}
            className="grid-line" />
        ))}
        {[1, 2].map((v) => (
          <line key={`gy${v}`}
            x1={xToSvg(xMin)} y1={yToSvg(v)} x2={xToSvg(xMax)} y2={yToSvg(v)}
            className="grid-line" />
        ))}

        {/* Linear reference (y=x) */}
        <path
          d={`M ${xToSvg(xMin)} ${yToSvg(xMin)} L ${xToSvg(xMax)} ${yToSvg(xMax)}`}
          className="reference-line"
        />

        {/* Main curve */}
        <path d={pathD} className="activation-curve" />

        {/* Annotation dots */}
        {annotations.map((a, i) => {
          const y = fn(a.x);
          return (
            <g key={i}>
              <circle cx={xToSvg(a.x)} cy={yToSvg(y)} r={4} className="annotation-dot" />
              <text
                x={xToSvg(a.x)}
                y={yToSvg(y) + (a.side === "below" ? 18 : -12)}
                className="annotation-text"
              >
                {a.label}
              </text>
            </g>
          );
        })}

        {/* Axis labels */}
        <text x={xToSvg(xMax) - 10} y={yToSvg(0) - 8} className="axis-label">x</text>
        <text x={xToSvg(0) + 8} y={yToSvg(yMax) + 14} className="axis-label">f(x)</text>
      </svg>
    </div>
  );
}
