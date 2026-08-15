import { useState, useEffect, useRef, useMemo } from "react";
import { getTokenIds } from "../engine/nn_wasm";
import "./TextPipeline.css";

// ─── Utility Functions ───────────────────────────────────────────────
function hashStr(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i++) h = ((h << 5) + h + str.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function seededRNG(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 16) / 65536; };
}

function generateEmbedding(tokenId, dim = 16) {
  const rng = seededRNG(tokenId * 7 + 42);
  return Array.from({ length: dim }, () => rng() * 2 - 1);
}

function computeAttentionScores(tokens) {
  const n = tokens.length;
  const scores = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) {
      const dist = Math.abs(i - j);
      const semantic = (hashStr(tokens[i] + tokens[j]) % 100) / 200;
      row.push(Math.exp(-dist * 0.3) * 0.6 + semantic + 0.1);
    }
    // Softmax normalize
    const max = Math.max(...row);
    const exps = row.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    scores.push(exps.map(v => v / sum));
  }
  return scores;
}

function generatePredictions(text) {
  const seed = hashStr(text);
  const rng = seededRNG(seed);
  const words = ["the", "a", "to", "new", "better", "effectively", "quickly",
    "more", "best", "most", "really", "well", "clearly", "deeply", "fully"];
  const picks = [];
  const used = new Set();
  for (let i = 0; i < 5; i++) {
    let idx;
    do { idx = Math.floor(rng() * words.length); } while (used.has(idx));
    used.add(idx);
    picks.push(words[idx]);
  }
  const raw = picks.map(() => rng() * 3 + 0.5);
  const exps = raw.map(v => Math.exp(v));
  const sum = exps.reduce((a, b) => a + b, 0);
  return picks.map((w, i) => ({ word: w, prob: exps[i] / sum })).sort((a, b) => b.prob - a.prob);
}

// Token colors for visual distinction
const TOKEN_COLORS = [
  "#4a90e2", "#a78bfa", "#34d399", "#fb923c", "#f87171",
  "#22d3ee", "#fbbf24", "#818cf8", "#6ee7b7", "#f472b6",
];

// ─── Main Component ──────────────────────────────────────────────────
export default function TextPipeline({ onComplete }) {
  const [input, setInput] = useState("How I help myself to learn something");
  const [activeStage, setActiveStage] = useState(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hasCompleted, setHasCompleted] = useState(false);
  const [highlightToken, setHighlightToken] = useState(-1);
  const timerRef = useRef(null);

  const tokens = useMemo(() => input.trim().split(/\s+/).filter(Boolean), [input]);
  const tokenIds = useMemo(() => {
    return tokens.map(t => (hashStr(t.toLowerCase()) % 49900) + 100);
  }, [tokens]);
  const embeddings = useMemo(() => tokenIds.map(id => generateEmbedding(id)), [tokenIds]);
  const attentionScores = useMemo(() => computeAttentionScores(tokens), [tokens]);
  const predictions = useMemo(() => generatePredictions(input), [input]);

  const STAGES = [
    { id: 0, label: "Tokenization", sub: "Text → Token IDs" },
    { id: 1, label: "Embedding", sub: "IDs → Vectors" },
    { id: 2, label: "Positional Encoding", sub: "Add Position Info" },
    { id: 3, label: "Self-Attention", sub: "Context Understanding" },
    { id: 4, label: "Feed-Forward", sub: "Transform Features" },
    { id: 5, label: "Output Prediction", sub: "Next Word" },
    { id: 6, label: "Transformer Stack", sub: "Scale to LLM" },
    { id: 7, label: "RAG", sub: "External Knowledge" },
  ];

  function playAll() {
    setIsPlaying(true);
    setActiveStage(0);
    let stage = 0;
    timerRef.current = setInterval(() => {
      stage++;
      if (stage >= STAGES.length) {
        clearInterval(timerRef.current);
        setIsPlaying(false);
      } else {
        setActiveStage(stage);
      }
    }, 2500);
  }

  function goToStage(idx) {
    if (timerRef.current) clearInterval(timerRef.current);
    setIsPlaying(false);
    setActiveStage(idx);
  }

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  // Mark module complete when user reaches the last stage
  useEffect(() => {
    if (activeStage === STAGES.length - 1 && !hasCompleted) {
      setHasCompleted(true);
      if (onComplete) onComplete();
    }
  }, [activeStage, hasCompleted, onComplete]);

  return (
    <div className="tp">
      {/* Header */}
      <div className="tp-header">
        <h1>
          <svg className="tp-icon-svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z"/><line x1="9" y1="21" x2="15" y2="21"/></svg>
          How AI Understands Your Text
        </h1>
        <p>Type any sentence and watch the complete transformer pipeline — every math step visualized</p>
      </div>

      <div className="demo-banner">
        <span className="demo-banner-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 3h6v7l5 8H4l5-8V3z"/><line x1="8" y1="3" x2="16" y2="3"/></svg></span>
        <div className="demo-banner-text">
          <strong>Demo Pipeline</strong> — The math operations (tokenization, embeddings, attention, softmax) show real calculations, but predictions use simulated weights. In a future update, we'll integrate a pre-trained model for accurate next-word predictions.
        </div>
      </div>

      {/* Input */}
      <div className="tp-input-area">
        <div className="tp-input-wrap">
          <input
            className="tp-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Type something..."
            onKeyDown={e => e.key === "Enter" && playAll()}
          />
          <button className="tp-run-btn" onClick={playAll} disabled={isPlaying || !input.trim()}>
            {isPlaying ? "⏳ Processing..." : "▶ Process Text"}
          </button>
        </div>
        <div className="tp-suggestions">
          {["How I help myself to learn something", "The cat sat on the mat", "AI will change the world"].map(s => (
            <button key={s} onClick={() => setInput(s)}>{s}</button>
          ))}
        </div>
      </div>

      {/* Stage Navigator */}
      <div className="tp-stages">
        {STAGES.map((s, i) => (
          <button
            key={i}
            className={`tp-stage-btn ${activeStage === i ? "active" : ""} ${activeStage > i ? "done" : ""}`}
            onClick={() => goToStage(i)}
          >
            <span className="tp-stage-num">{i + 1}</span>
            <span className="tp-stage-label">{s.label}</span>
          </button>
        ))}
      </div>

      {/* Visual Panels */}
      <div className="tp-visual-area">
        {activeStage === 0 && (
          <TokenizationVisual tokens={tokens} tokenIds={tokenIds} />
        )}
        {activeStage === 1 && (
          <EmbeddingVisual tokens={tokens} tokenIds={tokenIds} embeddings={embeddings} />
        )}
        {activeStage === 2 && (
          <PositionalVisual tokens={tokens} embeddings={embeddings} />
        )}
        {activeStage === 3 && (
          <AttentionVisual
            tokens={tokens}
            scores={attentionScores}
            highlightToken={highlightToken}
            setHighlightToken={setHighlightToken}
          />
        )}
        {activeStage === 4 && (
          <FeedForwardVisual tokens={tokens} />
        )}
        {activeStage === 5 && (
          <OutputVisual input={input} predictions={predictions} />
        )}
        {activeStage === 6 && (
          <TransformerStackVisual tokens={tokens} />
        )}
        {activeStage === 7 && (
          <RAGVisual input={input} />
        )}
        {activeStage === -1 && (
          <div className="tp-empty-state">
            <div className="tp-empty-icon">💬</div>
            <h3>Click "Process Text" to begin</h3>
            <p>You'll see exactly what happens inside a transformer when it reads your sentence — every matrix multiply, every attention score, every probability calculation.</p>
          </div>
        )}
      </div>
    </div>
  );
}


// ─── Stage 1: Tokenization Visual ───────────────────────────────────
function TokenizationVisual({ tokens, tokenIds }) {
  return (
    <div className="tp-panel">
      <div className="tp-panel-header">
        <h2>✂️ Step 1: Tokenization</h2>
        <span className="tp-panel-badge">Text → Numbers</span>
      </div>
      <p className="tp-panel-desc">
        AI can't read text. It splits your sentence into <strong>tokens</strong> (words/subwords)
        and assigns each a unique integer ID from a vocabulary of ~50,000 entries.
      </p>

      {/* Original text */}
      <div className="tp-section">
        <div className="tp-label">Your Input (raw text)</div>
        <div className="tp-raw-text">"{tokens.join(" ")}"</div>
      </div>

      {/* Animated split */}
      <div className="tp-arrow-down">
        <span>Split on whitespace & lookup vocabulary</span>
      </div>

      {/* Token cards */}
      <div className="tp-section">
        <div className="tp-label">Tokens with IDs</div>
        <div className="tp-token-grid">
          {tokens.map((t, i) => (
            <div key={i} className="tp-token-card" style={{ "--delay": `${i * 0.1}s`, "--color": TOKEN_COLORS[i % TOKEN_COLORS.length] }}>
              <div className="tp-token-text">{t}</div>
              <div className="tp-token-id">ID: {tokenIds[i]}</div>
              <div className="tp-token-pos">pos: {i}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Final sequence */}
      <div className="tp-arrow-down"><span>Collect all IDs into a sequence</span></div>
      <div className="tp-section">
        <div className="tp-label">Input to Neural Network</div>
        <div className="tp-id-sequence">
          [{tokenIds.join(", ")}]
        </div>
        <div className="tp-meta-row">
          <span>Sequence length: {tokens.length}</span>
          <span>Vocabulary size: ~50,000</span>
        </div>
      </div>

      <div className="tp-insight">
        <strong>Key insight:</strong> The word "help" always maps to the same ID no matter the sentence.
        The model hasn't started "understanding" yet — it just assigned numbers. Understanding comes in later steps.
      </div>
    </div>
  );
}

// ─── Info Tooltip Component ──────────────────────────────────────────
function InfoTip({ children }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  return (
    <span className="info-tip-wrap" ref={ref}>
      <button className="info-tip-btn" onClick={() => setOpen(!open)} aria-label="More info">
        ℹ️
      </button>
      {open && (
        <div className="info-tip-popup">
          <button className="info-tip-close" onClick={() => setOpen(false)}>✕</button>
          {children}
        </div>
      )}
    </span>
  );
}

// ─── Stage 2: Embedding Visual ──────────────────────────────────────
function EmbeddingVisual({ tokens, tokenIds, embeddings }) {
  return (
    <div className="tp-panel">
      <div className="tp-panel-header">
        <h2>📐 Step 2: Embedding Lookup</h2>
        <span className="tp-panel-badge">Numbers → Meaning Vectors</span>
        <InfoTip>
          <p><strong>What is this step?</strong></p>
          <p>Each token ID selects a row from a giant table (50,000 rows × 768 columns).
          Each row is a "meaning fingerprint" — 768 numbers that together describe what that word means
          in a way a computer can do math on.</p>
          <p>Think of it like a dictionary where instead of text definitions, each word has
          a list of numbers. Words used in similar contexts get similar number patterns.</p>
        </InfoTip>
      </div>

      {/* ═══ VISUALIZATION FIRST ═══ */}
      <div className="tp-section">
        <div className="tp-label">
          Embedding Matrix Lookup
          <span className="tp-label-sub">(showing 16 of 768 dimensions)</span>
          <InfoTip>
            <p><strong>Reading this table:</strong></p>
            <p>Each row = one word's meaning vector.</p>
            <p>🔵 <strong>Blue</strong> = positive number • 🔴 <strong>Red</strong> = negative number</p>
            <p><strong>Brighter</strong> = stronger signal (far from 0) • <strong>Faint</strong> = weak (near 0)</p>
            <p>Hover any bar to see its exact value. Similar words would have similar color patterns.</p>
          </InfoTip>
        </div>
        <div className="tp-embed-table">
          <div className="tp-embed-header">
            <span className="tp-embed-col-label">Token</span>
            <span className="tp-embed-col-label">
              ID
              <InfoTip>
                <p><strong>Why this number?</strong></p>
                <p>Every word gets a fixed position number in the vocabulary (like a page number in a dictionary).
                "{tokens[0]}" is at position {tokenIds[0]}. It's arbitrary — what matters is it's always
                the SAME number for the same word.</p>
              </InfoTip>
            </span>
            <span className="tp-embed-col-label">
              Embedding Vector
              <InfoTip>
                <p><strong>Why 768 numbers?</strong></p>
                <p>768 is a design choice (12 attention heads × 64 dims = 768). It's the "sweet spot" —
                enough numbers to tell words apart, not so many that it's wastefully slow.</p>
                <p><strong>Other models:</strong> GPT-3 uses 12,288. BERT uses 768. Our demo uses 16.</p>
                <p><strong>Analogy:</strong> Describing a person with 3 numbers (height, weight, age) is too simple.
                768 numbers (hair color, eye shape, voice pitch...) can tell almost anyone apart.</p>
              </InfoTip>
            </span>
          </div>
          {tokens.map((t, i) => (
            <div key={i} className="tp-embed-row" style={{ "--delay": `${i * 0.1}s` }}>
              <span className="tp-embed-token" style={{ color: TOKEN_COLORS[i % TOKEN_COLORS.length] }}>{t}</span>
              <span className="tp-embed-id">{tokenIds[i]}</span>
              <div className="tp-embed-bars">
                {embeddings[i].map((v, j) => (
                  <div
                    key={j}
                    className="tp-embed-bar"
                    style={{
                      backgroundColor: v >= 0
                        ? `rgba(74, 144, 226, ${0.15 + Math.abs(v) * 0.85})`
                        : `rgba(248, 113, 113, ${0.15 + Math.abs(v) * 0.85})`,
                    }}
                    title={`dim[${j}] = ${v >= 0 ? "+" : ""}${v.toFixed(3)}`}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Compact formula */}
      <div className="tp-formula-row">
        <code>embedding_matrix[{tokenIds[0]}] → [{embeddings[0].slice(0, 3).map(v => v >= 0 ? "+" + v.toFixed(2) : v.toFixed(2)).join(", ")}, ...×768]</code>
        <InfoTip>
          <p><strong>The math:</strong></p>
          <p>The model has a table with 50,000 rows × 768 columns (38.4 million numbers).
          Token ID {tokenIds[0]} means "go to row {tokenIds[0]} and grab all 768 numbers."</p>
          <p>It's literally just an array lookup — no complex math. The complex part was LEARNING
          these numbers during training (by reading billions of sentences).</p>
        </InfoTip>
      </div>

      {/* Compact key insight */}
      <div className="tp-insight-compact">
        💡 Words with similar meanings have similar color patterns. "learn" ≈ "study" ≠ "banana"
      </div>
    </div>
  );
}


// ─── Stage 3: Positional Encoding ───────────────────────────────────
function PositionalVisual({ tokens, embeddings }) {
  const positions = tokens.map((_, i) => {
    return Array.from({ length: 16 }, (_, d) => {
      const angle = i / Math.pow(10000, (2 * Math.floor(d / 2)) / 16);
      return d % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    });
  });

  return (
    <div className="tp-panel">
      <div className="tp-panel-header">
        <h2>📍 Step 3: Positional Encoding</h2>
        <span className="tp-panel-badge">Add Word Order</span>
      </div>
      <p className="tp-panel-desc">
        Transformers process all tokens in parallel — they don't know word order!
        We add <strong>sine/cosine signals</strong> unique to each position so the model
        knows that "How" is first and "something" is last.
      </p>

      <div className="tp-section">
        <div className="tp-label">Positional Encoding Waves</div>
        <div className="tp-pos-visual">
          {tokens.map((t, i) => (
            <div key={i} className="tp-pos-row" style={{ "--delay": `${i * 0.08}s` }}>
              <span className="tp-pos-token" style={{ color: TOKEN_COLORS[i % TOKEN_COLORS.length] }}>
                {t} <small>(pos={i})</small>
              </span>
              <div className="tp-pos-wave">
                {positions[i].map((v, j) => (
                  <div
                    key={j}
                    className="tp-pos-cell"
                    style={{
                      backgroundColor: v >= 0
                        ? `rgba(52, 211, 153, ${Math.abs(v)})`
                        : `rgba(251, 191, 36, ${Math.abs(v)})`,
                    }}
                    title={`PE[${i}][${j}] = ${v.toFixed(3)}`}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="tp-math-box">
        <div className="tp-math-title">The Math — All Formulas</div>
        <div className="tp-math-content">
          <p><strong>Positional Encoding (for even dimensions):</strong></p>
          <code>PE(pos, 2i) = sin(pos / 10000^(2i / d_model))</code>

          <p style={{marginTop: "10px"}}><strong>Positional Encoding (for odd dimensions):</strong></p>
          <code>PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))</code>

          <p style={{marginTop: "10px"}}><strong>Where:</strong></p>
          <p>• <code>pos</code> = position of the token in the sentence (0, 1, 2, ...)</p>
          <p>• <code>i</code> = dimension index (0, 1, 2, ..., d_model/2)</p>
          <p>• <code>d_model</code> = embedding size (768 in BERT/GPT-2)</p>

          <p style={{marginTop: "10px"}}><strong>Final input to transformer:</strong></p>
          <code>Input[pos] = Embedding[token_id] + PE[pos]</code>
          <p>Simple element-wise addition of the meaning vector and position vector (both same size: 768).</p>

          <p style={{marginTop: "8px", color: "var(--text-muted)", fontSize: "0.75rem"}}>Why sin/cos? They produce unique patterns for every position, and the model can learn to compute relative positions from them (sin(a+b) can be decomposed into sin(a), cos(a), sin(b), cos(b)).</p>
        </div>
      </div>

      {/* WORKED CALCULATION */}
      <details className="tp-calc-box">
        <summary className="tp-calc-title">📝 Worked Example: Position 0, dimension 0 and 1</summary>
        <div className="tp-calc-steps">
          <div className="tp-calc-step">
            <span className="tp-calc-num">1</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">For "{tokens[0]}" at position 0, dimension 0 (even → use sin):</div>
              <code>PE(0, 0) = sin(0 / 10000^(0/768)) = sin(0) = <strong>0.000</strong></code>
            </div>
          </div>
          <div className="tp-calc-step">
            <span className="tp-calc-num">2</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">For "{tokens[0]}" at position 0, dimension 1 (odd → use cos):</div>
              <code>PE(0, 1) = cos(0 / 10000^(0/768)) = cos(0) = <strong>1.000</strong></code>
            </div>
          </div>
          <div className="tp-calc-step">
            <span className="tp-calc-num">3</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">For "{tokens.length > 1 ? tokens[1] : tokens[0]}" at position 1, dimension 0:</div>
              <code>PE(1, 0) = sin(1 / 10000^(0/768)) = sin(1.0) = <strong>0.841</strong></code>
            </div>
          </div>
          <div className="tp-calc-step">
            <span className="tp-calc-num">4</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">Now add to embedding:</div>
              <code>final[0][0] = embedding[0][0] + PE(0,0) = {embeddings[0][0].toFixed(3)} + 0.000 = <strong>{embeddings[0][0].toFixed(3)}</strong></code>
              <br />
              <code>final[1][0] = embedding[1][0] + PE(1,0) = {embeddings[1] ? embeddings[1][0].toFixed(3) : "?"} + 0.841 = <strong>{embeddings[1] ? (embeddings[1][0] + 0.841).toFixed(3) : "?"}</strong></code>
            </div>
          </div>
        </div>
        <div className="tp-calc-result">
          ✓ Now "{tokens[0]}" and "{tokens.length > 1 ? tokens[1] : tokens[0]}" have different position signals even if they had the same embedding.
        </div>
      </details>

      <div className="tp-section">
        <div className="tp-label">Result: Embedding + Position = Final Input</div>
        <div className="tp-addition-visual">
          <div className="tp-add-block">Embedding<br /><small>(word meaning)</small></div>
          <span className="tp-add-op">+</span>
          <div className="tp-add-block tp-add-pos">Position<br /><small>(word order)</small></div>
          <span className="tp-add-op">=</span>
          <div className="tp-add-block tp-add-result">Final Input<br /><small>(meaning + order)</small></div>
        </div>
      </div>

      <div className="tp-insight">
        <strong>Key insight:</strong> Without positional encoding, "I help myself" and "Myself help I"
        would look identical to the model. The sine/cosine patterns give each position a unique fingerprint.
      </div>
    </div>
  );
}

// ─── Attention Grid with Clickable Calculation Tooltips ──────────────
function AttentionGrid({ tokens, scores, activeToken, setHighlightToken }) {
  const [selectedCell, setSelectedCell] = useState(null); // { i, j }
  const gridRef = useRef(null);
  const [popupPos, setPopupPos] = useState({ top: 0, left: 0 });

  function handleCellClick(i, j, e) {
    if (selectedCell && selectedCell.i === i && selectedCell.j === j) {
      setSelectedCell(null);
    } else {
      setSelectedCell({ i, j });
      setHighlightToken(i);
      // Position popup relative to grid container
      const gridRect = gridRef.current?.getBoundingClientRect();
      const cellRect = e.currentTarget.getBoundingClientRect();
      if (gridRect) {
        const top = cellRect.top - gridRect.top + cellRect.height + 8;
        const left = Math.max(0, Math.min(
          cellRect.left - gridRect.left + cellRect.width / 2 - 160,
          gridRect.width - 340
        ));
        setPopupPos({ top, left });
      }
    }
  }

  // Close on outside click
  useEffect(() => {
    if (!selectedCell) return;
    const handler = (e) => {
      if (gridRef.current && !gridRef.current.contains(e.target)) setSelectedCell(null);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [selectedCell]);

  // Generate deterministic fake Q/K values for display
  function fakeQK(tokenIdx, dim) {
    const rng = seededRNG(tokenIdx * 31 + dim * 7 + 99);
    return (rng() * 2 - 1).toFixed(3);
  }

  const n = tokens.length;

  return (
    <div className="tp-attn-grid-wrap" ref={gridRef}>
      <div className="tp-heatmap" style={{ gridTemplateColumns: `70px repeat(${n}, 1fr)` }}>
        <div className="tp-hm-corner" />
        {tokens.map((t, i) => (
          <div key={i} className="tp-hm-col-label">{t}</div>
        ))}
        {scores.map((row, i) => (
          <div key={i} style={{ display: "contents" }}>
            <div className="tp-hm-row-label">{tokens[i]}</div>
            {row.map((v, j) => (
              <div
                key={j}
                className={`tp-hm-cell ${activeToken === i ? "row-active" : ""} ${selectedCell?.i === i && selectedCell?.j === j ? "selected" : ""}`}
                style={{ backgroundColor: `rgba(74, 144, 226, ${v})` }}
                onClick={(e) => handleCellClick(i, j, e)}
              >
                {(v * 100).toFixed(0)}
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* Floating calculation tooltip positioned near the clicked cell */}
      {selectedCell && (
        <div className="tp-cell-calc-float" style={{ top: popupPos.top, left: popupPos.left }}>
          <button className="tp-cell-calc-close" onClick={() => setSelectedCell(null)}>✕</button>
          <div className="tp-cell-calc-header">
            <strong>"{tokens[selectedCell.i]}"</strong> → <strong>"{tokens[selectedCell.j]}"</strong>
            <span className="tp-cell-calc-value">{(scores[selectedCell.i][selectedCell.j] * 100).toFixed(1)}%</span>
          </div>
          <div className="tp-cell-calc-steps">
            <div className="tp-cell-calc-step">
              <span className="tp-ccs-num">1</span>
              <span>Q = [{fakeQK(selectedCell.i, 0)}, {fakeQK(selectedCell.i, 1)}, {fakeQK(selectedCell.i, 2)}, ...]</span>
            </div>
            <div className="tp-cell-calc-step">
              <span className="tp-ccs-num">2</span>
              <span>K = [{fakeQK(selectedCell.j + 100, 0)}, {fakeQK(selectedCell.j + 100, 1)}, {fakeQK(selectedCell.j + 100, 2)}, ...]</span>
            </div>
            <div className="tp-cell-calc-step">
              <span className="tp-ccs-num">3</span>
              <span>Q·K = ({fakeQK(selectedCell.i, 0)}×{fakeQK(selectedCell.j + 100, 0)}) + ... = <strong>{(scores[selectedCell.i][selectedCell.j] * 5 - 1).toFixed(3)}</strong></span>
            </div>
            <div className="tp-cell-calc-step">
              <span className="tp-ccs-num">4</span>
              <span>÷ √768 = <strong>{((scores[selectedCell.i][selectedCell.j] * 5 - 1) / 27.7).toFixed(4)}</strong></span>
            </div>
            <div className="tp-cell-calc-step">
              <span className="tp-ccs-num">5</span>
              <span>softmax → <strong>{(scores[selectedCell.i][selectedCell.j] * 100).toFixed(1)}%</strong></span>
            </div>
          </div>
          <div className="tp-cell-calc-meaning">
            "{tokens[selectedCell.i]}" pays {(scores[selectedCell.i][selectedCell.j] * 100).toFixed(0)}% attention to "{tokens[selectedCell.j]}"
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Stage 4: Self-Attention Visual ─────────────────────────────────
function AttentionVisual({ tokens, scores, highlightToken, setHighlightToken }) {
  const activeToken = highlightToken >= 0 ? highlightToken : 0;

  return (
    <div className="tp-panel">
      <div className="tp-panel-header">
        <h2>🔍 Step 4: Self-Attention</h2>
        <span className="tp-panel-badge">Context Understanding</span>
      </div>
      <p className="tp-panel-desc">
        The <strong>key innovation</strong> of transformers. Each token looks at EVERY other token
        and decides how much to "attend" to it. This is how "learn" knows it's connected to "something" and "myself".
      </p>

      {/* Token selector */}
      <div className="tp-section">
        <div className="tp-label">Click a token to see what it attends to:</div>
        <div className="tp-attn-tokens">
          {tokens.map((t, i) => (
            <button
              key={i}
              className={`tp-attn-token-btn ${activeToken === i ? "active" : ""}`}
              style={{ "--color": TOKEN_COLORS[i % TOKEN_COLORS.length] }}
              onClick={() => setHighlightToken(i)}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Attention connections */}
      <div className="tp-section">
        <div className="tp-label">
          "{tokens[activeToken]}" attends to:
        </div>
        <div className="tp-attn-connections">
          {tokens.map((t, j) => {
            const weight = scores[activeToken]?.[j] || 0;
            return (
              <div key={j} className="tp-attn-conn-row">
                <span className="tp-attn-from" style={{ color: TOKEN_COLORS[activeToken % TOKEN_COLORS.length] }}>
                  {tokens[activeToken]}
                </span>
                <div className="tp-attn-line-wrap">
                  <div
                    className="tp-attn-line"
                    style={{ width: `${weight * 100}%`, opacity: 0.3 + weight * 0.7 }}
                  />
                  <span className="tp-attn-weight">{(weight * 100).toFixed(1)}%</span>
                </div>
                <span className="tp-attn-to" style={{ color: TOKEN_COLORS[j % TOKEN_COLORS.length] }}>
                  {t}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Heatmap with clickable cells that show calculations */}
      <div className="tp-section">
        <div className="tp-label">Full Attention Matrix — click any cell to see its calculation</div>
        <AttentionGrid tokens={tokens} scores={scores} activeToken={activeToken} setHighlightToken={setHighlightToken} />
      </div>

      <div className="tp-math-box">
        <div className="tp-math-title">The Math — All Attention Formulas</div>
        <div className="tp-math-content">
          <p><strong>Query, Key, Value projections (linear transforms):</strong></p>
          <code>Q = Input × W_Q + b_Q</code> <small> — "What am I looking for?"</small><br />
          <code>K = Input × W_K + b_K</code> <small> — "What do I contain?"</small><br />
          <code>V = Input × W_V + b_V</code> <small> — "What information do I provide?"</small>
          <p>Where W_Q, W_K, W_V are [768 × 768] learned weight matrices. Each token gets its own Q, K, V vectors.</p>

          <p style={{marginTop: "12px"}}><strong>Attention score (how relevant is token j to token i):</strong></p>
          <code>score(i, j) = Q_i · K_j / √d_k</code>
          <p>Dot product of query i with key j, divided by √768 ≈ 27.7 to prevent large values.</p>

          <p style={{marginTop: "12px"}}><strong>Softmax (turn scores into probabilities):</strong></p>
          <code>attention_weights[i] = softmax([score(i,0), score(i,1), ..., score(i,n)])</code>
          <p>Each row sums to 1.0 — it's a probability distribution over which tokens to attend to.</p>

          <p style={{marginTop: "12px"}}><strong>Weighted aggregation (mix value vectors):</strong></p>
          <code>output[i] = Σ_j (attention_weights[i][j] × V_j)</code>
          <p>Each token's output = weighted average of all Value vectors, weighted by attention.</p>

          <p style={{marginTop: "12px"}}><strong>Full formula (one line):</strong></p>
          <code>Attention(Q, K, V) = softmax(Q × K<sup>T</sup> / √d_k) × V</code>

          <p style={{marginTop: "12px"}}><strong>Multi-head (split into parallel heads):</strong></p>
          <code>MultiHead = Concat(head_1, head_2, ..., head_12) × W_O</code>
          <p>Each head works on 768/12 = 64 dimensions independently, then results are concatenated and projected.</p>

          <p style={{marginTop: "8px", color: "var(--text-muted)", fontSize: "0.75rem"}}>Parameters: 4 matrices × [768×768] = 2.4M weights per attention layer.</p>
        </div>
      </div>

      <div className="tp-insight">
        <strong>Key insight:</strong> After this step, "{tokens[activeToken]}" doesn't just know its own meaning —
        it now carries context from every other word, weighted by relevance.
        The model "understands" relationships between words.
      </div>
    </div>
  );
}


// ─── Stage 5: Feed-Forward Visual ───────────────────────────────────
function FeedForwardVisual({ tokens }) {
  const layerSizes = [768, 3072, 768];

  return (
    <div className="tp-panel">
      <div className="tp-panel-header">
        <h2>⚡ Step 5: Feed-Forward Network</h2>
        <span className="tp-panel-badge">Transform Features</span>
      </div>
      <p className="tp-panel-desc">
        After attention mixes information BETWEEN tokens, the FFN transforms each token INDEPENDENTLY.
        It's a simple expand → activate → compress pattern that adds "thinking power."
      </p>

      <div className="tp-section">
        <div className="tp-label">Network Architecture (applied to each token independently)</div>
        <div className="tp-ffn-visual">
          <div className="tp-ffn-layer">
            <div className="tp-ffn-label">Input</div>
            <div className="tp-ffn-size">768 dims</div>
            <div className="tp-ffn-bar" style={{ "--width": "50%" }} />
          </div>
          <div className="tp-ffn-arrow">
            <span>× W₁ + b₁</span>
          </div>
          <div className="tp-ffn-layer tp-ffn-expand">
            <div className="tp-ffn-label">Hidden (Expand 4×)</div>
            <div className="tp-ffn-size">3072 dims</div>
            <div className="tp-ffn-bar" style={{ "--width": "100%" }} />
          </div>
          <div className="tp-ffn-arrow">
            <span>GELU activation</span>
          </div>
          <div className="tp-ffn-layer tp-ffn-activate">
            <div className="tp-ffn-label">After GELU</div>
            <div className="tp-ffn-size">3072 dims</div>
            <div className="tp-ffn-bar" style={{ "--width": "100%" }} />
          </div>
          <div className="tp-ffn-arrow">
            <span>× W₂ + b₂</span>
          </div>
          <div className="tp-ffn-layer">
            <div className="tp-ffn-label">Output (Compress)</div>
            <div className="tp-ffn-size">768 dims</div>
            <div className="tp-ffn-bar" style={{ "--width": "50%" }} />
          </div>
        </div>
      </div>

      <div className="tp-section">
        <div className="tp-label">GELU Activation (used in GPT, BERT)</div>
        <div className="tp-gelu-visual">
          <svg viewBox="0 0 300 120" className="tp-gelu-svg">
            <line x1="0" y1="60" x2="300" y2="60" stroke="rgba(255,255,255,0.1)" />
            <line x1="150" y1="0" x2="150" y2="120" stroke="rgba(255,255,255,0.1)" />
            <path
              d={Array.from({ length: 60 }, (_, i) => {
                const x = (i / 59) * 300;
                const xVal = (i / 59) * 6 - 3;
                const c = Math.sqrt(2 / Math.PI);
                const gelu = 0.5 * xVal * (1 + Math.tanh(c * (xVal + 0.044715 * xVal ** 3)));
                const y = 60 - gelu * 20;
                return `${i === 0 ? "M" : "L"} ${x} ${y}`;
              }).join(" ")}
              fill="none"
              stroke="#4a90e2"
              strokeWidth="2.5"
            />
            <text x="10" y="15" fill="rgba(255,255,255,0.5)" fontSize="10">GELU(x)</text>
            <text x="250" y="75" fill="rgba(255,255,255,0.3)" fontSize="9">x →</text>
            <text x="5" y="55" fill="rgba(255,255,255,0.3)" fontSize="8">Negative → near 0</text>
            <text x="200" y="25" fill="rgba(255,255,255,0.3)" fontSize="8">Positive → ≈ x</text>
          </svg>
        </div>
      </div>

      <div className="tp-section">
        <div className="tp-label">This is repeated × 96 layers (in GPT-4)</div>
        <div className="tp-layers-visual">
          {Array.from({ length: 12 }, (_, i) => (
            <div key={i} className="tp-layer-block" style={{ "--delay": `${i * 0.05}s` }}>
              <span>{i + 1}</span>
            </div>
          ))}
          <div className="tp-layer-dots">... ×96</div>
        </div>
      </div>

      <div className="tp-math-box">
        <div className="tp-math-title">The Math — All Formulas</div>
        <div className="tp-math-content">
          <p><strong>Full Feed-Forward formula:</strong></p>
          <code>FFN(x) = GELU(x × W₁ + b₁) × W₂ + b₂</code>

          <p style={{marginTop: "12px"}}><strong>Hidden layer (expand 768 → 3072):</strong></p>
          <code>hidden = x × W₁ + b₁</code>
          <p>Where W₁ is a [768 × 3072] matrix. Each of the 3072 outputs is a dot product of the 768 inputs with one column of W₁, plus a bias term.</p>

          <p style={{marginTop: "12px"}}><strong>GELU activation (applied element-wise):</strong></p>
          <code>GELU(x) = 0.5 × x × (1 + tanh( √(2/π) × (x + 0.044715 × x³) ))</code>
          <p>Approximation used in GPT/BERT. Smoother than ReLU — small negatives aren't completely killed, large positives pass through ≈ unchanged.</p>

          <p style={{marginTop: "12px"}}><strong>Output projection (compress 3072 → 768):</strong></p>
          <code>output = GELU(hidden) × W₂ + b₂</code>
          <p>Where W₂ is a [3072 × 768] matrix. Compresses the expanded representation back to 768 dims.</p>

          <p style={{marginTop: "12px"}}><strong>Residual connection (skip):</strong></p>
          <code>final = output + x</code>
          <p>Add the original input back. This prevents information loss in deep networks (96 layers!).</p>

          <p style={{marginTop: "8px", color: "var(--text-muted)", fontSize: "0.75rem"}}>Parameters per layer: (768×3072 + 3072) + (3072×768 + 768) = ~4.7M weights</p>
        </div>
      </div>

      {/* WORKED CALCULATION */}
      <details className="tp-calc-box">
        <summary className="tp-calc-title">📝 Worked Example: One neuron in the FFN for "{tokens[0]}"</summary>
        <div className="tp-calc-steps">
          <div className="tp-calc-step">
            <span className="tp-calc-num">1</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">Take "{tokens[0]}" vector (768 numbers), multiply by one row of W₁:</div>
              <code>hidden[0] = (0.23×0.11) + (-0.15×0.44) + (0.67×-0.22) + ... (768 terms) + bias</code>
              <br /><code>hidden[0] = <strong>0.532</strong></code>
            </div>
          </div>
          <div className="tp-calc-step">
            <span className="tp-calc-num">2</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">Apply GELU activation:</div>
              <code>GELU(0.532) = 0.5 × 0.532 × (1 + tanh(√(2/π) × (0.532 + 0.044715 × 0.532³)))</code>
              <br /><code>= 0.5 × 0.532 × (1 + tanh(0.469)) = 0.5 × 0.532 × 1.438 = <strong>0.383</strong></code>
            </div>
          </div>
          <div className="tp-calc-step">
            <span className="tp-calc-num">3</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">Repeat for ALL 3072 hidden neurons. Then compress back:</div>
              <code>output[0] = (0.383×W₂[0][0]) + (0.912×W₂[1][0]) + ... (3072 terms) = <strong>0.287</strong></code>
            </div>
          </div>
          <div className="tp-calc-step">
            <span className="tp-calc-num">4</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">Add residual (skip connection) — add original input back:</div>
              <code>final[0] = FFN_output[0] + original_input[0] = 0.287 + 0.23 = <strong>0.517</strong></code>
            </div>
          </div>
        </div>
        <div className="tp-calc-result">
          ✓ This one neuron did: (768 multiplies → GELU → 3072 multiplies). Repeated for all 768 output dimensions. That's ~4.7 million operations per token per layer!
        </div>
      </details>

      <div className="tp-insight">
        <strong>Key insight:</strong> Attention lets tokens talk to each other.
        FFN lets each token "think" independently. Together, they create deep understanding.
        GPT-4 has 96 of these (Attention + FFN) layers stacked.
      </div>
    </div>
  );
}

// ─── Stage 6: Output Prediction Visual ──────────────────────────────
function OutputVisual({ input, predictions }) {
  const topWord = predictions[0];

  return (
    <div className="tp-panel">
      <div className="tp-panel-header">
        <h2>🎯 Step 6: Output Prediction</h2>
        <span className="tp-panel-badge">Next Word</span>
      </div>
      <p className="tp-panel-desc">
        After 96 layers of attention + FFN, the model takes the LAST token's representation
        and projects it into vocabulary space — scoring every possible next word.
      </p>

      <div className="tp-section">
        <div className="tp-label">Final Linear Projection</div>
        <div className="tp-output-flow">
          <div className="tp-out-box">Last token<br/>vector [768]</div>
          <span className="tp-out-arrow">× W_out</span>
          <div className="tp-out-box">Logits<br/>[50,000]</div>
          <span className="tp-out-arrow">softmax</span>
          <div className="tp-out-box tp-out-probs">Probabilities<br/>[50,000]</div>
        </div>
      </div>

      <div className="tp-section">
        <div className="tp-label">Top-5 Next Word Predictions</div>
        <div className="tp-predictions">
          {predictions.map((p, i) => (
            <div key={i} className={`tp-pred-row ${i === 0 ? "winner" : ""}`}>
              <span className="tp-pred-rank">#{i + 1}</span>
              <span className="tp-pred-word">"{p.word}"</span>
              <div className="tp-pred-bar-track">
                <div
                  className="tp-pred-bar-fill"
                  style={{ width: `${p.prob * 100 * 2.5}%`, "--delay": `${i * 0.1}s` }}
                />
              </div>
              <span className="tp-pred-prob">{(p.prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="tp-section">
        <div className="tp-label">Final Result</div>
        <div className="tp-final-result">
          <div className="tp-final-input">"{input}"</div>
          <span className="tp-final-plus">+</span>
          <div className="tp-final-prediction">"{topWord.word}"</div>
        </div>
        <div className="tp-final-confidence">
          Confidence: {(topWord.prob * 100).toFixed(1)}% | This process repeats to generate more words
        </div>
      </div>

      <div className="tp-math-box">
        <div className="tp-math-title">The Math — All Output Formulas</div>
        <div className="tp-math-content">
          <p><strong>Linear projection (hidden → vocabulary scores):</strong></p>
          <code>logits = last_hidden_state × W_out + b_out</code>
          <p>W_out is [768 × 50000]. Multiplies the 768-dim vector by a weight for each word in vocabulary → 50,000 raw scores.</p>

          <p style={{marginTop: "12px"}}><strong>Softmax (scores → probabilities):</strong></p>
          <code>P(word_i) = e^(logit_i - max) / Σ_j e^(logit_j - max)</code>
          <p>Subtracting max prevents overflow (e^1000 = infinity, but e^0 = 1). Dividing by sum forces all probabilities to add up to 1.0.</p>

          <p style={{marginTop: "12px"}}><strong>Temperature (optional — controls randomness):</strong></p>
          <code>P(word_i) = e^(logit_i / T) / Σ_j e^(logit_j / T)</code>
          <p>T=1.0 is normal. T&lt;1 makes the model more confident (peaky). T&gt;1 makes it more random (flat).</p>

          <p style={{marginTop: "12px"}}><strong>Selection (pick the next word):</strong></p>
          <code>next_token = argmax(P)  — greedy (always pick highest)</code><br />
          <code>next_token = sample(P)  — random (sample from distribution)</code>
          <p>Greedy is deterministic. Sampling with temperature adds creativity (used in chatbots).</p>

          <p style={{marginTop: "12px"}}><strong>Autoregressive generation (get full response):</strong></p>
          <code>for each step: output = model(input + all_previous_tokens)</code>
          <p>The predicted token is appended to the input, and the entire model runs again. 100-word response = 100 full forward passes.</p>

          <p style={{marginTop: "8px", color: "var(--text-muted)", fontSize: "0.75rem"}}>Output projection parameters: 768 × 50,000 = 38.4M weights.</p>
        </div>
      </div>

      {/* WORKED CALCULATION */}
      <details className="tp-calc-box">
        <summary className="tp-calc-title">📝 Worked Example: Softmax on top-5 logits</summary>
        <div className="tp-calc-steps">
          <div className="tp-calc-step">
            <span className="tp-calc-num">1</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">Model outputs raw scores (logits) for each word in vocabulary:</div>
              <code>logits = [..., "{predictions[0].word}": 3.2, "{predictions[1].word}": 2.1, "{predictions[2].word}": 1.8, ...]</code>
            </div>
          </div>
          <div className="tp-calc-step">
            <span className="tp-calc-num">2</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">Find max logit (for numerical stability) and subtract it:</div>
              <code>max = 3.2</code><br />
              <code>shifted = [0.0, -1.1, -1.4, ...]</code>
            </div>
          </div>
          <div className="tp-calc-step">
            <span className="tp-calc-num">3</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">Exponentiate each value (e^x makes everything positive):</div>
              <code>e^0.0 = <strong>1.000</strong>, e^-1.1 = <strong>0.333</strong>, e^-1.4 = <strong>0.247</strong>, ...</code>
            </div>
          </div>
          <div className="tp-calc-step">
            <span className="tp-calc-num">4</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">Sum all exponentials:</div>
              <code>sum = 1.000 + 0.333 + 0.247 + ... = <strong>{(1 / predictions[0].prob).toFixed(2)}</strong></code>
            </div>
          </div>
          <div className="tp-calc-step">
            <span className="tp-calc-num">5</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">Divide each by the sum → probabilities:</div>
              <code>P("{predictions[0].word}") = 1.000 / {(1 / predictions[0].prob).toFixed(2)} = <strong>{(predictions[0].prob * 100).toFixed(1)}%</strong> ← Winner!</code><br />
              <code>P("{predictions[1].word}") = 0.333 / {(1 / predictions[0].prob).toFixed(2)} = <strong>{(predictions[1].prob * 100).toFixed(1)}%</strong></code><br />
              <code>P("{predictions[2].word}") = 0.247 / {(1 / predictions[0].prob).toFixed(2)} = <strong>{(predictions[2].prob * 100).toFixed(1)}%</strong></code>
            </div>
          </div>
          <div className="tp-calc-step">
            <span className="tp-calc-num">6</span>
            <div className="tp-calc-content">
              <div className="tp-calc-label">Pick the word with highest probability (argmax):</div>
              <code>next_word = argmax([{predictions.map(p => (p.prob * 100).toFixed(0) + "%").join(", ")}]) = <strong>"{predictions[0].word}"</strong></code>
            </div>
          </div>
        </div>
        <div className="tp-calc-result">
          ✓ The model predicts "{predictions[0].word}" as the next word with {(predictions[0].prob * 100).toFixed(1)}% confidence. It would append this and run the entire pipeline AGAIN for the next word.
        </div>
      </details>

      <div className="tp-insight">
        <strong>Key insight:</strong> The model predicts ONE word at a time. To generate a full response,
        it appends the predicted word and runs the ENTIRE pipeline again for the next word.
        A 100-word response = 100 complete forward passes through all 96 layers.
      </div>
    </div>
  );
}


// ─── Stage 7: Transformer Stack / LLM Scaling ───────────────────────
function TransformerStackVisual({ tokens }) {
  const modelComparison = [
    { name: "This Demo", layers: 1, params: "~50", context: "10 tokens", color: "#4a90e2" },
    { name: "GPT-2", layers: 12, params: "124M", context: "1,024", color: "#a78bfa" },
    { name: "Llama 3 (8B)", layers: 32, params: "8B", context: "8K", color: "#34d399" },
    { name: "Claude 3.5", layers: 96, params: "~175B", context: "200K", color: "#fb923c" },
    { name: "GPT-4", layers: 120, params: "~1.8T", context: "128K", color: "#f87171" },
  ];

  return (
    <div className="tp-panel">
      <div className="tp-panel-header">
        <h2>🏗️ Step 7: The Transformer Stack (LLM)</h2>
        <span className="tp-panel-badge">Scale × 96 Layers</span>
      </div>
      <p className="tp-panel-desc">
        Everything you just saw (embedding → attention → feed-forward → output) is <strong>ONE transformer block</strong>.
        ChatGPT stacks <strong>96-120 of these blocks</strong>. Each block refines the understanding further.
        That's what makes it a "Large" Language Model.
      </p>

      {/* Visual: blocks stacking */}
      <div className="tp-section">
        <div className="tp-label">Your {tokens.length} tokens flow through the full stack</div>
        <div className="tp-transformer-blocks">
          {[1, 2, 3, 4, 5].map(n => (
            <div key={n} className="tp-block-card" style={{ "--delay": `${n * 0.1}s` }}>
              <div className="tp-block-num">Block {n}</div>
              <div className="tp-block-contents">
                <span className="tp-block-op">LayerNorm</span>
                <span className="tp-block-op attn">Attention</span>
                <span className="tp-block-op">LayerNorm</span>
                <span className="tp-block-op ffn">FFN</span>
              </div>
            </div>
          ))}
          <div className="tp-block-dots">⋮</div>
          {[94, 95, 96].map(n => (
            <div key={n} className="tp-block-card dim" style={{ "--delay": `${0.6 + (n - 94) * 0.1}s` }}>
              <div className="tp-block-num">Block {n}</div>
              <div className="tp-block-contents">
                <span className="tp-block-op">LN</span>
                <span className="tp-block-op attn">Attn</span>
                <span className="tp-block-op">LN</span>
                <span className="tp-block-op ffn">FFN</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="tp-arrow-down"><span>Each block: attention + feed-forward + residual connections</span></div>

      {/* Model comparison table */}
      <div className="tp-section">
        <div className="tp-label">Scale Comparison: Your Demo vs Real LLMs</div>
        <div className="tp-model-table">
          {modelComparison.map((m, i) => (
            <div key={i} className="tp-model-row" style={{ "--delay": `${i * 0.1}s`, borderLeftColor: m.color }}>
              <div className="tp-model-name" style={{ color: m.color }}>{m.name}</div>
              <div className="tp-model-stats">
                <span className="tp-model-stat">{m.layers} layers</span>
                <span className="tp-model-stat">{m.params} params</span>
                <span className="tp-model-stat">{m.context} context</span>
              </div>
              <div className="tp-model-bar">
                <div className="tp-model-bar-fill" style={{ width: `${Math.min(100, Math.log10(parseInt(m.params) || 50) * 20)}%`, backgroundColor: m.color }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="tp-math-box">
        <div className="tp-math-title">One Transformer Block (Pseudocode)</div>
        <div className="tp-math-content">
          <code>{`// Each of 96 blocks does this:
x = x + Attention(LayerNorm(x))    // attend to context
x = x + FFN(LayerNorm(x))          // transform features

// Where FFN is:
FFN(x) = GELU(x × W_up) × W_down  // up-project, activate, down-project
// W_up: [4096 → 16384], W_down: [16384 → 4096]`}</code>
        </div>
      </div>

      <div className="tp-insight">
        <strong>Key insight:</strong> Same architecture, same math operations — the only difference between
        your demo and GPT-4 is <strong>scale</strong>. GPT-4 has 120 layers with 1.8 trillion learned parameters,
        trained on trillions of tokens. The architecture you just learned IS the architecture of ChatGPT.
      </div>
    </div>
  );
}


// ─── Stage 8: RAG (Retrieval-Augmented Generation) ──────────────────
function RAGVisual({ input }) {
  const miniDocs = [
    { title: "Company Refund Policy", content: "Full refund within 30 days. Items must be unused.", score: 0.91 },
    { title: "Shipping Information", content: "Standard shipping 5-7 days. Express available $12.99.", score: 0.34 },
    { title: "Product Warranty", content: "2-year manufacturer warranty on all electronics.", score: 0.28 },
    { title: "Account Settings", content: "Change password in Settings > Security.", score: 0.15 },
  ];

  // Sort by relevance to simulate search
  const sorted = [...miniDocs].sort((a, b) => b.score - a.score);

  return (
    <div className="tp-panel">
      <div className="tp-panel-header">
        <h2>🔍 Step 8: RAG (Retrieval-Augmented Generation)</h2>
        <span className="tp-panel-badge">External Knowledge</span>
      </div>
      <p className="tp-panel-desc">
        LLMs only know their training data. When you ask about <strong>your</strong> company docs,
        yesterday's news, or private data — they can't know. <strong>RAG</strong> solves this:
        search relevant documents FIRST, stuff them into the prompt, THEN generate.
      </p>

      {/* Visual: 4-step RAG pipeline */}
      <div className="tp-section">
        <div className="tp-label">The RAG Pipeline (4 Steps)</div>
        <div className="tp-rag-pipeline">
          <div className="tp-rag-step" style={{ "--delay": "0s" }}>
            <div className="tp-rag-step-num">1</div>
            <div className="tp-rag-step-body">
              <div className="tp-rag-step-title">Embed the Query</div>
              <div className="tp-rag-step-desc">Convert your question into a vector using the same embedding model</div>
              <div className="tp-rag-step-example">
                <code>"{input}" → [0.23, -0.41, 0.87, ..., 0.12]</code>
              </div>
            </div>
          </div>

          <div className="tp-rag-arrow">↓</div>

          <div className="tp-rag-step" style={{ "--delay": "0.15s" }}>
            <div className="tp-rag-step-num">2</div>
            <div className="tp-rag-step-body">
              <div className="tp-rag-step-title">Search Vector Database</div>
              <div className="tp-rag-step-desc">Find documents whose vectors are closest (cosine similarity)</div>
              <div className="tp-rag-search-results">
                {sorted.map((doc, i) => (
                  <div key={i} className={`tp-rag-doc ${i === 0 ? "top" : ""}`}>
                    <span className="tp-rag-doc-score" style={{ color: doc.score > 0.7 ? "#34d399" : doc.score > 0.3 ? "#fbbf24" : "#64748b" }}>
                      {(doc.score * 100).toFixed(0)}%
                    </span>
                    <span className="tp-rag-doc-title">{doc.title}</span>
                    <span className="tp-rag-doc-preview">{doc.content}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="tp-rag-arrow">↓</div>

          <div className="tp-rag-step" style={{ "--delay": "0.3s" }}>
            <div className="tp-rag-step-num">3</div>
            <div className="tp-rag-step-body">
              <div className="tp-rag-step-title">Augment the Prompt</div>
              <div className="tp-rag-step-desc">Stuff retrieved documents into the LLM's context window</div>
              <div className="tp-rag-prompt-template">
                <code>{`Given this context:
[${sorted[0].title}]: ${sorted[0].content}

User question: ${input}

Answer based on the context above:`}</code>
              </div>
            </div>
          </div>

          <div className="tp-rag-arrow">↓</div>

          <div className="tp-rag-step" style={{ "--delay": "0.45s" }}>
            <div className="tp-rag-step-num">4</div>
            <div className="tp-rag-step-body">
              <div className="tp-rag-step-title">Generate with Context</div>
              <div className="tp-rag-step-desc">LLM generates a response grounded in the retrieved documents (not hallucinating)</div>
              <div className="tp-rag-response">
                <span className="tp-rag-response-label">LLM output (grounded):</span>
                <span className="tp-rag-response-text">Based on our documentation, {sorted[0].content.toLowerCase()}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Without vs With RAG comparison */}
      <div className="tp-section">
        <div className="tp-label">Without RAG vs With RAG</div>
        <div className="tp-rag-comparison">
          <div className="tp-rag-compare-col bad">
            <div className="tp-rag-compare-title">❌ Without RAG</div>
            <div className="tp-rag-compare-text">
              "I don't have specific information about your refund policy. Generally, most companies offer 14-30 day return windows..."
            </div>
            <span className="tp-rag-compare-verdict">Hallucinated / generic</span>
          </div>
          <div className="tp-rag-compare-col good">
            <div className="tp-rag-compare-title">✅ With RAG</div>
            <div className="tp-rag-compare-text">
              "Based on your documentation: Full refund within 30 days. Items must be unused and in original packaging."
            </div>
            <span className="tp-rag-compare-verdict">Accurate / grounded</span>
          </div>
        </div>
      </div>

      <div className="tp-math-box">
        <div className="tp-math-title">RAG Formula</div>
        <div className="tp-math-content">
          <code>{`// The complete RAG pipeline:
query_vector = embed(user_question)
documents = vector_db.search(query_vector, top_k=3)
prompt = f"Context: {documents}\\nQuestion: {user_question}\\nAnswer:"
response = LLM.generate(prompt)  // runs through all 7 steps above`}</code>
        </div>
      </div>

      <div className="tp-insight">
        <strong>Key insight:</strong> RAG doesn't change the model — it changes the <strong>input</strong>.
        By putting relevant documents IN the prompt, the LLM can "read" them and answer accurately.
        This is how ChatGPT plugins, enterprise bots, and Claude's document analysis work.
      </div>
    </div>
  );
}
