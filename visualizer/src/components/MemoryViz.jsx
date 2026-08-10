import { useState, useRef, useEffect, useCallback } from "react";
import "./MemoryViz.css";

const ARENA_SIZE = 600; // px width representing arena capacity
const ARENA_CAPACITY_BYTES = 16 * 1024 * 1024; // 16 MB

// Simulated allocations for a transformer layer
const INFERENCE_ALLOCS = [
  { name: "Q projection", size: 32 * 128 * 768 * 4, color: "#3b82f6" },
  { name: "K projection", size: 32 * 128 * 768 * 4, color: "#8b5cf6" },
  { name: "V projection", size: 32 * 128 * 768 * 4, color: "#6366f1" },
  { name: "Attn scores", size: 32 * 12 * 128 * 128 * 4, color: "#f59e0b" },
  { name: "Softmax out", size: 32 * 12 * 128 * 128 * 4, color: "#f97316" },
  { name: "Attn output", size: 32 * 128 * 768 * 4, color: "#22c55e" },
  { name: "FFN expand", size: 32 * 128 * 3072 * 4, color: "#ec4899" },
  { name: "FFN output", size: 32 * 128 * 768 * 4, color: "#14b8a6" },
];

function formatBytes(b) {
  if (b >= 1024 * 1024) return (b / 1024 / 1024).toFixed(1) + " MB";
  if (b >= 1024) return (b / 1024).toFixed(0) + " KB";
  return b + " B";
}

export default function MemoryViz() {
  const [allocations, setAllocations] = useState([]);
  const [offset, setOffset] = useState(0);
  const [peakUsage, setPeakUsage] = useState(0);
  const [allocCount, setAllocCount] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [mode, setMode] = useState("arena"); // "arena" or "malloc"
  const [mallocBlocks, setMallocBlocks] = useState([]);
  const [mallocFragmented, setMallocFragmented] = useState(0);
  const timerRef = useRef(null);

  const reset = useCallback(() => {
    setAllocations([]);
    setOffset(0);
    setAllocCount(0);
    setMallocBlocks([]);
    setMallocFragmented(0);
    setIsRunning(false);
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  const runArenaDemo = useCallback(() => {
    reset();
    setMode("arena");
    setIsRunning(true);
    let step = 0;
    let currentOffset = 0;
    const allocs = [];

    timerRef.current = setInterval(() => {
      if (step < INFERENCE_ALLOCS.length) {
        const alloc = INFERENCE_ALLOCS[step];
        const pxWidth = Math.max(8, (alloc.size / ARENA_CAPACITY_BYTES) * ARENA_SIZE);
        const pxOffset = (currentOffset / ARENA_CAPACITY_BYTES) * ARENA_SIZE;

        allocs.push({ ...alloc, pxOffset, pxWidth, step });
        currentOffset += alloc.size;

        setAllocations([...allocs]);
        setOffset(currentOffset);
        setAllocCount(step + 1);
        setPeakUsage(currentOffset);
        step++;
      } else if (step === INFERENCE_ALLOCS.length) {
        // Reset animation
        setTimeout(() => {
          setAllocations([]);
          setOffset(0);
          setAllocCount(prev => prev); // keep count
        }, 800);
        step++;
      } else {
        setIsRunning(false);
        clearInterval(timerRef.current);
      }
    }, 400);
  }, [reset]);

  const runMallocDemo = useCallback(() => {
    reset();
    setMode("malloc");
    setIsRunning(true);
    let step = 0;
    const blocks = [];
    let fragCount = 0;

    timerRef.current = setInterval(() => {
      if (step < INFERENCE_ALLOCS.length * 2) {
        if (step < INFERENCE_ALLOCS.length) {
          // Allocate
          const alloc = INFERENCE_ALLOCS[step];
          const gap = Math.random() * 20 + 5; // Fragmentation gaps
          const lastEnd = blocks.length > 0 ? blocks[blocks.length - 1].end : 0;
          const pxWidth = Math.max(8, (alloc.size / ARENA_CAPACITY_BYTES) * ARENA_SIZE * 0.6);
          blocks.push({
            ...alloc,
            start: lastEnd + gap,
            end: lastEnd + gap + pxWidth,
            pxWidth,
            freed: false,
            step,
          });
          fragCount += gap;
        } else {
          // Free one by one (slow!)
          const freeIdx = step - INFERENCE_ALLOCS.length;
          if (freeIdx < blocks.length) {
            blocks[freeIdx].freed = true;
          }
        }
        setMallocBlocks([...blocks]);
        setMallocFragmented(fragCount);
        setAllocCount(Math.min(step + 1, INFERENCE_ALLOCS.length));
        step++;
      } else {
        setIsRunning(false);
        clearInterval(timerRef.current);
      }
    }, 300);
  }, [reset]);

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  return (
    <div className="memviz">
      <div className="memviz-header">
        <h1>
          <span className="memviz-icon">🧠</span>
          Memory Arena Visualizer
        </h1>
        <p>Watch how arena allocation works vs traditional malloc — and why it's 918× faster</p>
      </div>

      {/* Mode selector */}
      <div className="memviz-controls">
        <button className={`mv-btn ${mode === "arena" ? "active" : ""}`} onClick={runArenaDemo} disabled={isRunning}>
          ▶ Arena Allocator
        </button>
        <button className={`mv-btn mv-btn-red ${mode === "malloc" ? "active" : ""}`} onClick={runMallocDemo} disabled={isRunning}>
          ▶ malloc/free
        </button>
        <button className="mv-btn mv-btn-gray" onClick={reset}>Reset</button>
      </div>

      {/* Arena Visualization */}
      {mode === "arena" && (
        <div className="memviz-arena-section">
          <div className="arena-label">Arena (16 MB pre-allocated, one contiguous block)</div>
          <div className="arena-bar">
            <div className="arena-bg">
              {allocations.map((a, i) => (
                <div
                  key={i}
                  className="arena-block"
                  style={{
                    left: `${a.pxOffset}px`,
                    width: `${a.pxWidth}px`,
                    backgroundColor: a.color,
                    animationDelay: `${i * 0.05}s`,
                  }}
                  title={`${a.name}: ${formatBytes(a.size)}`}
                />
              ))}
              {/* Pointer indicator */}
              <div
                className="arena-pointer"
                style={{ left: `${(offset / ARENA_CAPACITY_BYTES) * ARENA_SIZE}px` }}
              >
                <div className="pointer-arrow">▼</div>
                <div className="pointer-label">offset</div>
              </div>
            </div>
          </div>

          <div className="arena-legend">
            {allocations.map((a, i) => (
              <div key={i} className="legend-item">
                <span className="legend-dot" style={{ background: a.color }} />
                <span className="legend-name">{a.name}</span>
                <span className="legend-size">{formatBytes(a.size)}</span>
              </div>
            ))}
          </div>

          {/* How it works */}
          <div className="arena-explain">
            <div className="explain-card">
              <div className="explain-title">Allocate</div>
              <div className="explain-body">
                <code>ptr = base + offset; offset += size;</code>
                <div className="explain-cost">Cost: ~4 nanoseconds (1 addition)</div>
              </div>
            </div>
            <div className="explain-card">
              <div className="explain-title">Reset (free all)</div>
              <div className="explain-body">
                <code>offset = 0;</code>
                <div className="explain-cost">Cost: ~1 nanosecond (1 assignment)</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Malloc Visualization */}
      {mode === "malloc" && (
        <div className="memviz-malloc-section">
          <div className="arena-label">Heap (fragmented, each alloc is a syscall)</div>
          <div className="arena-bar">
            <div className="arena-bg malloc-bg">
              {mallocBlocks.map((b, i) => (
                <div
                  key={i}
                  className={`malloc-block ${b.freed ? "freed" : ""}`}
                  style={{
                    left: `${b.start}px`,
                    width: `${b.pxWidth}px`,
                    backgroundColor: b.freed ? "#334155" : b.color,
                  }}
                  title={`${b.name}${b.freed ? " (freed)" : ""}`}
                />
              ))}
              {/* Fragmentation gaps shown as red areas */}
            </div>
          </div>

          <div className="malloc-explain">
            <div className="explain-card warn">
              <div className="explain-title">malloc()</div>
              <div className="explain-body">
                <code>ptr = find_free_block(size); update_free_list();</code>
                <div className="explain-cost">Cost: ~50-200 nanoseconds (lock + search + syscall)</div>
              </div>
            </div>
            <div className="explain-card warn">
              <div className="explain-title">free()</div>
              <div className="explain-body">
                <code>mark_free(ptr); maybe_coalesce(); maybe_return_to_OS();</code>
                <div className="explain-cost">Cost: ~50-100 nanoseconds per block</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Stats comparison */}
      <div className="memviz-stats">
        <div className="mvs-card">
          <div className="mvs-label">Allocations</div>
          <div className="mvs-value">{allocCount}</div>
        </div>
        <div className="mvs-card">
          <div className="mvs-label">Peak Usage</div>
          <div className="mvs-value">{formatBytes(peakUsage)}</div>
        </div>
        <div className="mvs-card">
          <div className="mvs-label">Fragmentation</div>
          <div className="mvs-value">{mode === "arena" ? "0%" : `${mallocFragmented > 0 ? "~12%" : "0%"}`}</div>
        </div>
        <div className="mvs-card highlight">
          <div className="mvs-label">Arena Speed</div>
          <div className="mvs-value">918×</div>
          <div className="mvs-sub">faster than malloc</div>
        </div>
      </div>

      {/* Conceptual comparison */}
      <div className="memviz-comparison">
        <h3>Why Arena Wins for Inference</h3>
        <div className="compare-grid">
          <div className="compare-col">
            <div className="compare-header arena-h">Arena</div>
            <ul>
              <li>Pre-allocate once at startup</li>
              <li>Each alloc = bump a pointer (O(1))</li>
              <li>Zero fragmentation (contiguous)</li>
              <li>Free all = reset pointer (O(1))</li>
              <li>Cache-friendly (sequential access)</li>
            </ul>
          </div>
          <div className="compare-col">
            <div className="compare-header malloc-h">malloc/free</div>
            <ul>
              <li>Syscall per allocation (lock + search)</li>
              <li>Free-list traversal to find space</li>
              <li>Fragmentation grows over time</li>
              <li>Each free = update metadata</li>
              <li>Cache-hostile (scattered pointers)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
