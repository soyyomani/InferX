import { useState, useMemo, useCallback } from "react";
import "./GraphOptimizer.css";

// ─── Graph Definitions ───────────────────────────────────────────────

const EXAMPLE_GRAPHS = {
  mlp: {
    name: "MLP (2-Layer)",
    description: "Input → MatMul → ReLU → MatMul → Softmax",
    nodes: [
      { id: "input", label: "Input", type: "input", shape: "[32, 784]", x: 50, y: 200 },
      { id: "w1", label: "Weights₁", type: "input", shape: "[784, 128]", x: 50, y: 350 },
      { id: "mm1", label: "MatMul", type: "matmul", shape: "[32, 128]", x: 250, y: 270 },
      { id: "relu", label: "ReLU", type: "relu", shape: "[32, 128]", x: 450, y: 270 },
      { id: "w2", label: "Weights₂", type: "input", shape: "[128, 10]", x: 450, y: 400 },
      { id: "mm2", label: "MatMul", type: "matmul", shape: "[32, 10]", x: 650, y: 320 },
      { id: "softmax", label: "Softmax", type: "softmax", shape: "[32, 10]", x: 850, y: 320 },
    ],
    edges: [
      { from: "input", to: "mm1" },
      { from: "w1", to: "mm1" },
      { from: "mm1", to: "relu" },
      { from: "relu", to: "mm2" },
      { from: "w2", to: "mm2" },
      { from: "mm2", to: "softmax" },
    ],
  },
  transformer: {
    name: "Transformer Block",
    description: "Q/K/V Projections → Attention → FFN with GELU",
    nodes: [
      { id: "input", label: "Input", type: "input", shape: "[32, 128, 768]", x: 50, y: 250 },
      { id: "wq", label: "W_Q", type: "input", shape: "[768, 768]", x: 150, y: 100 },
      { id: "wk", label: "W_K", type: "input", shape: "[768, 768]", x: 150, y: 250 },
      { id: "wv", label: "W_V", type: "input", shape: "[768, 768]", x: 150, y: 400 },
      { id: "mmq", label: "MatMul_Q", type: "matmul", shape: "[32, 128, 768]", x: 320, y: 100 },
      { id: "mmk", label: "MatMul_K", type: "matmul", shape: "[32, 128, 768]", x: 320, y: 250 },
      { id: "mmv", label: "MatMul_V", type: "matmul", shape: "[32, 128, 768]", x: 320, y: 400 },
      { id: "attn", label: "Attention", type: "softmax", shape: "[32, 128, 768]", x: 520, y: 250 },
      { id: "wff", label: "W_FFN", type: "input", shape: "[768, 3072]", x: 620, y: 400 },
      { id: "mmff", label: "MatMul", type: "matmul", shape: "[32, 128, 3072]", x: 720, y: 300 },
      { id: "gelu", label: "GELU", type: "gelu", shape: "[32, 128, 3072]", x: 900, y: 300 },
    ],
    edges: [
      { from: "input", to: "mmq" }, { from: "wq", to: "mmq" },
      { from: "input", to: "mmk" }, { from: "wk", to: "mmk" },
      { from: "input", to: "mmv" }, { from: "wv", to: "mmv" },
      { from: "mmq", to: "attn" }, { from: "mmk", to: "attn" }, { from: "mmv", to: "attn" },
      { from: "attn", to: "mmff" }, { from: "wff", to: "mmff" },
      { from: "mmff", to: "gelu" },
    ],
  },
};

// ─── Optimizer Logic (mirrors C++ optimizer.h) ───────────────────────

function applyFusion(nodes, edges) {
  const fusedNodes = [...nodes];
  const fusedEdges = [...edges];
  const fusions = [];
  const removedIds = new Set();

  // Find MatMul → Activation patterns (single consumer)
  for (const node of nodes) {
    if (node.type !== "matmul") continue;

    // Count consumers of this matmul
    const consumers = edges.filter(e => e.from === node.id);
    if (consumers.length !== 1) continue;

    const consumerId = consumers[0].to;
    const consumer = nodes.find(n => n.id === consumerId);
    if (!consumer) continue;

    // Check if consumer is an activation
    if (consumer.type === "relu" || consumer.type === "gelu") {
      const fusedType = consumer.type === "relu" ? "fused_matmul_relu" : "fused_matmul_gelu";
      const fusedLabel = consumer.type === "relu" ? "FusedMatMul+ReLU" : "FusedMatMul+GELU";

      fusions.push({
        matmul: node.id,
        activation: consumer.id,
        fusedType,
        fusedLabel,
        memorySaved: `${node.shape} intermediate eliminated`,
      });

      removedIds.add(node.id);
    }
  }

  // Apply fusions: replace activation nodes with fused ops
  const result = fusedNodes.map(n => {
    const fusion = fusions.find(f => f.activation === n.id);
    if (fusion) {
      return { ...n, type: fusion.fusedType, label: fusion.fusedLabel, fused: true };
    }
    if (removedIds.has(n.id)) {
      return { ...n, dead: true };
    }
    return n;
  });

  // Rewire edges: fused node now takes matmul's inputs directly
  const newEdges = [];
  for (const edge of fusedEdges) {
    if (removedIds.has(edge.to)) {
      // Edge going to dead matmul → redirect to the fused activation
      const fusion = fusions.find(f => f.matmul === edge.to);
      if (fusion) {
        newEdges.push({ ...edge, to: fusion.activation });
      }
    } else if (removedIds.has(edge.from)) {
      // Skip edges from dead matmul to its activation (now internal)
      continue;
    } else {
      newEdges.push(edge);
    }
  }

  return { nodes: result, edges: newEdges, fusions, removedIds };
}

function applyDeadNodeElimination(nodes, edges) {
  // Find nodes with no consumers that aren't outputs
  const outputNodes = new Set(["softmax", "gelu", "relu"]);
  const hasConsumer = new Set(edges.map(e => e.from));
  const dead = nodes.filter(n =>
    !n.dead && !hasConsumer.has(n.id) && !outputNodes.has(n.type) &&
    !nodes.some(other => other.id !== n.id && edges.some(e => e.from === n.id))
  );
  return dead.map(n => n.id);
}

// ─── Main Component ──────────────────────────────────────────────────
export default function GraphOptimizer() {
  const [selectedGraph, setSelectedGraph] = useState("mlp");
  const [optimized, setOptimized] = useState(false);
  const [animating, setAnimating] = useState(false);
  const [step, setStep] = useState(0); // 0=original, 1=highlight pattern, 2=fused

  const graph = EXAMPLE_GRAPHS[selectedGraph];

  const optimizedResult = useMemo(() => {
    return applyFusion(graph.nodes, graph.edges);
  }, [graph]);

  const handleOptimize = useCallback(() => {
    setAnimating(true);
    setStep(1); // Highlight fusion candidates

    setTimeout(() => {
      setStep(2); // Apply fusion
      setOptimized(true);
      setAnimating(false);
    }, 1500);
  }, []);

  const handleReset = () => {
    setOptimized(false);
    setStep(0);
    setAnimating(false);
  };

  const handleGraphChange = (id) => {
    setSelectedGraph(id);
    handleReset();
  };

  const displayNodes = step >= 2 ? optimizedResult.nodes : graph.nodes;
  const displayEdges = step >= 2 ? optimizedResult.edges : graph.edges;

  // Stats
  const originalOps = graph.nodes.filter(n => n.type !== "input").length;
  const fusedOps = optimizedResult.nodes.filter(n => !n.dead && n.type !== "input").length;
  const numFusions = optimizedResult.fusions.length;

  return (
    <div className="graph-opt">
      {/* Header */}
      <div className="graph-header">
        <h1>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3"/><path d="M12 2v4m0 12v4m10-10h-4M6 12H2m15.07-5.07l-2.83 2.83M9.76 14.24l-2.83 2.83m11.14 0l-2.83-2.83M9.76 9.76L6.93 6.93"/>
          </svg>
          Graph Optimizer
        </h1>
        <p>Watch operator fusion transform the computational graph — the core of AI compilers</p>
      </div>

      {/* Graph Selector */}
      <div className="graph-selector">
        {Object.entries(EXAMPLE_GRAPHS).map(([id, g]) => (
          <button
            key={id}
            className={`graph-sel-btn ${selectedGraph === id ? "active" : ""}`}
            onClick={() => handleGraphChange(id)}
          >
            {g.name}
          </button>
        ))}
      </div>

      {/* Controls */}
      <div className="graph-controls">
        <button
          className="btn-optimize"
          onClick={handleOptimize}
          disabled={optimized || animating}
        >
          {animating ? "⚡ Fusing..." : optimized ? "✓ Optimized" : "⚡ Run Optimizer"}
        </button>
        <button className="btn-reset" onClick={handleReset} disabled={!optimized && step === 0}>
          Reset
        </button>
        <div className="graph-desc">{graph.description}</div>
      </div>

      {/* Graph Visualization */}
      <div className="graph-canvas-wrap">
        <svg className="graph-svg" viewBox="0 0 1000 500" preserveAspectRatio="xMidYMid meet">
          {/* Edges */}
          {displayEdges.map((edge, i) => {
            const fromNode = displayNodes.find(n => n.id === edge.from);
            const toNode = displayNodes.find(n => n.id === edge.to);
            if (!fromNode || !toNode || fromNode.dead || toNode.dead) return null;

            const x1 = fromNode.x + 60;
            const y1 = fromNode.y + 20;
            const x2 = toNode.x;
            const y2 = toNode.y + 20;

            const isHighlighted = step === 1 &&
              optimizedResult.fusions.some(f =>
                (edge.from === f.matmul && edge.to === f.activation));

            return (
              <line
                key={i}
                x1={x1} y1={y1} x2={x2} y2={y2}
                className={`graph-edge ${isHighlighted ? "highlight" : ""}`}
                markerEnd="url(#arrowhead)"
              />
            );
          })}

          {/* Arrow marker */}
          <defs>
            <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill="#64748b" />
            </marker>
          </defs>

          {/* Nodes */}
          {displayNodes.map((node) => {
            if (node.dead) return null;

            const isCandidate = step === 1 &&
              (optimizedResult.fusions.some(f => f.matmul === node.id || f.activation === node.id));

            const nodeClass = [
              "graph-node",
              `type-${node.type}`,
              node.fused ? "fused" : "",
              isCandidate ? "candidate" : "",
            ].filter(Boolean).join(" ");

            return (
              <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
                <rect
                  className={nodeClass}
                  width="120" height="45"
                  rx="6"
                />
                <text className="node-label" x="60" y="18" textAnchor="middle">
                  {node.label}
                </text>
                <text className="node-shape" x="60" y="35" textAnchor="middle">
                  {node.shape}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Optimization Stats */}
      <div className="opt-stats">
        <div className="opt-stat-card">
          <span className="stat-label">Original Ops</span>
          <span className="stat-value">{originalOps}</span>
        </div>
        <div className="opt-stat-card">
          <span className="stat-label">After Fusion</span>
          <span className="stat-value highlight">{fusedOps}</span>
        </div>
        <div className="opt-stat-card">
          <span className="stat-label">Fusions Applied</span>
          <span className="stat-value">{numFusions}</span>
        </div>
        <div className="opt-stat-card">
          <span className="stat-label">Memory Saved</span>
          <span className="stat-value">1 intermediate</span>
        </div>
      </div>

      {/* Fusion Explanation */}
      {optimizedResult.fusions.length > 0 && (
        <div className="fusion-explain">
          <h3>Fusion Passes Applied</h3>
          {optimizedResult.fusions.map((f, i) => (
            <div key={i} className="fusion-card">
              <div className="fusion-pattern">
                <span className="fuse-from">MatMul</span>
                <span className="fuse-arrow">+</span>
                <span className="fuse-from">{f.fusedType.includes("relu") ? "ReLU" : "GELU"}</span>
                <span className="fuse-arrow">→</span>
                <span className="fuse-to">{f.fusedLabel}</span>
              </div>
              <div className="fusion-benefit">
                <strong>Why:</strong> Eliminates intermediate tensor write+read ({f.memorySaved}).
                The activation is applied while data is still in CPU registers.
              </div>
              <div className="fusion-ref">
                Same optimization used by: TensorRT, oneDNN, XLA, cuDNN
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
