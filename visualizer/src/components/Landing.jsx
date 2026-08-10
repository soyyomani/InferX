import "./Landing.css";

export default function Landing({ onNavigate }) {
  return (
    <div className="home">
      {/* ═══ Hero ═══ */}
      <section className="home-hero">
        <p className="home-tag">C++20 Inference Engine</p>
        <h1>Built from first principles.</h1>
        <p className="home-lead">
          Tensor engine, SIMD kernels, memory allocators, graph compiler,
          quantization, thread pool — zero external ML dependencies.
        </p>
        <div className="home-cta">
          <button onClick={() => onNavigate("arch")}>Explore the architecture</button>
          <button className="sec" onClick={() => onNavigate("mnist")}>Try live demo</button>
        </div>
      </section>

      {/* ═══ Numbers ═══ */}
      <section className="home-numbers">
        <div><span>22</span>GFLOPS</div>
        <div><span>918×</span>vs malloc</div>
        <div><span>216+</span>tests</div>
        <div><span>100%</span>MNIST</div>
        <div><span>C++20</span>standard</div>
      </section>

      {/* ═══ Two Halves ═══ */}
      <section className="home-split">
        <div className="split-col">
          <h2>The algorithms</h2>
          <p className="split-desc">What AI models actually compute — visualized step by step.</p>
          <ul className="split-list">
            <li onClick={() => onNavigate("text")}>
              <span className="sl-name">Text Pipeline</span>
              <span className="sl-detail">tokenize → embed → attend → predict</span>
            </li>
            <li onClick={() => onNavigate("image")}>
              <span className="sl-name">Vision Pipeline</span>
              <span className="sl-detail">pixels → conv → pool → classify</span>
            </li>
            <li onClick={() => onNavigate("mnist")}>
              <span className="sl-name">Live Inference</span>
              <span className="sl-detail">draw a digit → real-time prediction</span>
            </li>
            <li onClick={() => onNavigate("math")}>
              <span className="sl-name">Math Lab</span>
              <span className="sl-detail">matmul, softmax, relu, attention</span>
            </li>
          </ul>
        </div>

        <div className="split-col">
          <h2>The systems</h2>
          <p className="split-desc">The C++ infrastructure that makes it fast enough to deploy.</p>
          <ul className="split-list">
            <li onClick={() => onNavigate("kernels")}>
              <span className="sl-name">SIMD Kernels</span>
              <span className="sl-detail">ARM NEON, cache tiling, 4×4 micro-kernel</span>
            </li>
            <li onClick={() => onNavigate("memory")}>
              <span className="sl-name">Memory Arena</span>
              <span className="sl-detail">bump allocator, zero-malloc inference</span>
            </li>
            <li onClick={() => onNavigate("graph")}>
              <span className="sl-name">Graph Compiler</span>
              <span className="sl-detail">topo sort, dead code elim, op fusion</span>
            </li>
            <li onClick={() => onNavigate("quantize")}>
              <span className="sl-name">Quantization</span>
              <span className="sl-detail">float32 → int8, 4× compression</span>
            </li>
            <li onClick={() => onNavigate("threads")}>
              <span className="sl-name">Thread Pool</span>
              <span className="sl-detail">task DAG, parallel_for, ~50ns dispatch</span>
            </li>
            <li onClick={() => onNavigate("tensor")}>
              <span className="sl-name">Tensor Engine</span>
              <span className="sl-detail">zero-copy reshape, broadcast, iterator</span>
            </li>
          </ul>
        </div>
      </section>

      {/* ═══ How they connect ═══ */}
      <section className="home-connect">
        <p className="connect-text">
          <code>Attention(Q, K, V)</code> is an algorithm.{" "}
          Making it run at <strong>7,485 images/sec</strong> with{" "}
          <strong>552 bytes</strong> peak memory is systems engineering.
        </p>
        <p className="connect-sub">This project is both.</p>
      </section>

      {/* ═══ Footer line ═══ */}
      <section className="home-footer-line">
        <span>Open source</span>
        <span>·</span>
        <span>MIT license</span>
        <span>·</span>
        <span>No external ML deps</span>
        <span>·</span>
        <span>Apple Silicon optimized</span>
      </section>
    </div>
  );
}
