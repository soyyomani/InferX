import "./Landing.css";

const FEATURES = [
  {
    id: "text",
    icon: "💬",
    title: "Text AI Pipeline",
    desc: "Type a prompt and watch AI process it — tokenization, embeddings, attention, prediction. Every math step visualized.",
    tags: ["Tokenizer", "Attention", "Softmax"],
    color: "accent",
  },
  {
    id: "image",
    icon: "🖼",
    title: "Vision AI Pipeline",
    desc: "Upload an image and see how neural networks extract features — convolutions, pooling, classification. Real pixel math.",
    tags: ["Convolution", "Pooling", "Classification"],
    color: "purple",
  },
  {
    id: "math",
    icon: "∑",
    title: "Math Lab",
    desc: "Explore individual operations interactively — matrix multiply, softmax, ReLU, GELU. See every calculation step by step.",
    tags: ["MatMul", "Softmax", "ReLU", "GELU"],
    color: "orange",
  },
  {
    id: "tensor",
    icon: "▦",
    title: "Tensor Playground",
    desc: "Build tensors from scratch, apply operations, and watch memory layout in 3D. Powered by real C++ compiled to WebAssembly.",
    tags: ["Shape", "Stride", "WASM", "3D"],
    color: "cyan",
  },
];

const MATH_CONCEPTS = [
  { name: "Tokenization", desc: "How text becomes numbers" },
  { name: "Embeddings", desc: "How numbers become meaning" },
  { name: "Attention", desc: "How words understand context" },
  { name: "Matrix Multiply", desc: "The core computation" },
  { name: "Softmax", desc: "How scores become probabilities" },
  { name: "Backpropagation", desc: "How networks learn (coming soon)" },
];

export default function Landing({ onNavigate }) {
  return (
    <div className="landing">
      {/* Hero */}
      <section className="hero">
        <div className="hero-badge">Open Source • C++20 • WebAssembly</div>
        <h1 className="hero-title">
          See How AI <span className="gradient-text">Actually Thinks</span>
        </h1>
        <p className="hero-subtitle">
          When you send a prompt to ChatGPT or upload an image to a classifier,
          what happens inside? InferX visualizes every single math operation — from raw
          input to final prediction — powered by real C++ running in your browser.
        </p>
        <div className="hero-actions">
          <button className="btn btn-primary btn-lg" onClick={() => onNavigate("text")}>
            Try Text Pipeline →
          </button>
          <button className="btn btn-secondary btn-lg" onClick={() => onNavigate("math")}>
            Explore Math
          </button>
        </div>
        <div className="hero-stats">
          <div className="stat">
            <span className="stat-value">6</span>
            <span className="stat-label">Math Operations</span>
          </div>
          <div className="stat">
            <span className="stat-value">C++20</span>
            <span className="stat-label">Backend Engine</span>
          </div>
          <div className="stat">
            <span className="stat-value">WASM</span>
            <span className="stat-label">Browser Runtime</span>
          </div>
          <div className="stat">
            <span className="stat-value">Step-by-Step</span>
            <span className="stat-label">Visualization</span>
          </div>
        </div>
      </section>

      {/* Feature Cards */}
      <section className="features-section">
        <h2 className="section-title">What You Can Explore</h2>
        <p className="section-subtitle">
          Each module shows the real math that AI uses — no hand-waving, no black boxes
        </p>
        <div className="features-grid">
          {FEATURES.map((f) => (
            <div
              key={f.id}
              className={`feature-card feature-${f.color}`}
              onClick={() => onNavigate(f.id)}
            >
              <div className="feature-icon">{f.icon}</div>
              <h3 className="feature-title">{f.title}</h3>
              <p className="feature-desc">{f.desc}</p>
              <div className="feature-tags">
                {f.tags.map((t) => (
                  <span key={t} className="feature-tag">{t}</span>
                ))}
              </div>
              <span className="feature-arrow">→</span>
            </div>
          ))}
        </div>
      </section>

      {/* Math Concepts */}
      <section className="concepts-section">
        <h2 className="section-title">The Math Behind AI</h2>
        <p className="section-subtitle">
          Every AI model — GPT, DALL-E, Claude — uses these core operations
        </p>
        <div className="concepts-grid">
          {MATH_CONCEPTS.map((c) => (
            <div key={c.name} className="concept-card">
              <span className="concept-name">{c.name}</span>
              <span className="concept-desc">{c.desc}</span>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section className="how-section">
        <h2 className="section-title">How It Works</h2>
        <div className="how-steps">
          <div className="how-step">
            <div className="how-number">1</div>
            <div className="how-content">
              <h4>You provide input</h4>
              <p>Type text like "Hello world" or upload an image</p>
            </div>
          </div>
          <div className="how-connector" />
          <div className="how-step">
            <div className="how-number">2</div>
            <div className="how-content">
              <h4>C++ engine processes it</h4>
              <p>Real compiled C++ runs tokenization, embeddings, attention — recording every step</p>
            </div>
          </div>
          <div className="how-connector" />
          <div className="how-step">
            <div className="how-number">3</div>
            <div className="how-content">
              <h4>You see the math</h4>
              <p>Every matrix multiply, every softmax, every attention score — visualized step by step</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
