import { Typography, Button, Row, Col, Space, Divider, Progress, Tag } from "antd";
import {
  FunctionOutlined,
  RobotOutlined,
  EyeOutlined,
  EditOutlined,
  PlayCircleOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ArrowRightOutlined,
  LockOutlined,
  TrophyOutlined,
} from "@ant-design/icons";
import "./Landing.css";

const { Title, Paragraph, Text } = Typography;

const LEARNING_PATH = [
  {
    key: "math",
    title: "Math Lab",
    subtitle: "Master the building blocks",
    icon: <FunctionOutlined />,
    color: "#fbbf24",
    gradient: "linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)",
    time: "15 min",
    prereq: null,
  },
  {
    key: "textai",
    title: "Text AI",
    subtitle: "How ChatGPT thinks",
    icon: <RobotOutlined />,
    color: "#4a90e2",
    gradient: "linear-gradient(135deg, #4a90e2 0%, #3b82f6 100%)",
    time: "40 min",
    prereq: "Math Lab",
  },
  {
    key: "visionai",
    title: "Vision AI",
    subtitle: "How AI sees images",
    icon: <EyeOutlined />,
    color: "#34d399",
    gradient: "linear-gradient(135deg, #34d399 0%, #10b981 100%)",
    time: "20 min",
    prereq: "Text AI",
  },
  {
    key: "mnist",
    title: "Playground",
    subtitle: "Draw & predict live",
    icon: <EditOutlined />,
    color: "#a78bfa",
    gradient: "linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%)",
    time: "5 min",
    prereq: "Vision AI",
  },
];

export default function Landing({ onNavigate, visitedPages = [] }) {
  const completedSteps = LEARNING_PATH.filter(s => visitedPages.includes(s.key)).length;
  const nextStep = LEARNING_PATH.find(s => !visitedPages.includes(s.key));

  return (
    <div className="landing">
      {/* ═══ Hero ═══ */}
      <section className="landing-hero">
        <div className="hero-glow" />
        <div className="hero-orb hero-orb-1" />
        <div className="hero-orb hero-orb-2" />

        <div className="hero-content">
          <div className="hero-badge">
            <ThunderboltOutlined /> AI Education Platform
          </div>

          <h1 className="hero-title-custom">
            See how AI
            <span className="hero-gradient-text"> actually thinks.</span>
          </h1>

          <p className="hero-subtitle">
            Type a prompt. Watch every calculation. Understand transformers, attention, and neural networks — not by reading, but by <em>seeing them run</em>.
          </p>

          <div className="hero-cta-row">
            <Button
              type="primary"
              size="large"
              icon={<PlayCircleOutlined />}
              className="hero-btn-start"
              onClick={() => onNavigate(nextStep?.key || "math")}
            >
              {completedSteps === 0 ? "Start the Journey" : "Continue"}
            </Button>
            {completedSteps > 0 && (
              <div className="hero-progress-mini">
                <Progress
                  type="circle"
                  percent={Math.round((completedSteps / 4) * 100)}
                  size={40}
                  strokeColor="#4a90e2"
                  format={() => `${completedSteps}/4`}
                />
              </div>
            )}
          </div>
        </div>
      </section>

      {/* ═══ Learning Path — Visual Cards ═══ */}
      <section className="landing-path">
        <div className="path-header">
          <span className="path-header-num">4 modules</span>
          <span className="path-header-line" />
          <span className="path-header-label">from zero to live inference</span>
        </div>

        <div className="path-cards">
          {LEARNING_PATH.map((step, idx) => {
            const isCompleted = visitedPages.includes(step.key);
            const isNext = step.key === nextStep?.key;
            const isLocked = idx > 0 && !visitedPages.includes(LEARNING_PATH[idx - 1].key) && !isCompleted;

            return (
              <div key={step.key} className="path-card-wrapper">
                {/* Connector arrow */}
                {idx > 0 && <div className="path-connector"><ArrowRightOutlined /></div>}

                <div
                  className={`path-card-v2 ${isCompleted ? "completed" : ""} ${isNext ? "next" : ""} ${isLocked ? "locked" : ""}`}
                  onClick={() => !isLocked && onNavigate(step.key)}
                >
                  {/* Top colored strip */}
                  <div className="pc-top-strip" style={{ background: isLocked ? "#374151" : step.gradient }} />

                  {/* Number badge */}
                  <div className="pc-number" style={{ background: isCompleted ? "#34d399" : isLocked ? "#374151" : step.color }}>
                    {isCompleted ? <CheckCircleOutlined /> : isLocked ? <LockOutlined /> : idx + 1}
                  </div>

                  {/* Icon */}
                  <div className="pc-icon" style={{ color: isLocked ? "#4b5563" : step.color }}>
                    {step.icon}
                  </div>

                  {/* Text */}
                  <div className="pc-title">{step.title}</div>
                  <div className="pc-subtitle">{step.subtitle}</div>

                  {/* Footer */}
                  <div className="pc-footer">
                    <span className="pc-time">{step.time}</span>
                    {isCompleted && <span className="pc-status done">Done</span>}
                    {isNext && <span className="pc-status next-tag">Next</span>}
                    {isLocked && <span className="pc-status locked-tag">Locked</span>}
                  </div>

                  {/* CTA for next */}
                  {isNext && (
                    <button className="pc-cta" onClick={(e) => { e.stopPropagation(); onNavigate(step.key); }}>
                      Start <ArrowRightOutlined />
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* ═══ Quick Stats — visual, minimal ═══ */}
      <section className="landing-stats">
        <div className="stats-row">
          <div className="stat-item"><span className="stat-value">22</span><span className="stat-label">GFLOPS</span></div>
          <div className="stat-divider" />
          <div className="stat-item"><span className="stat-value">918x</span><span className="stat-label">vs malloc</span></div>
          <div className="stat-divider" />
          <div className="stat-item"><span className="stat-value">C++20</span><span className="stat-label">standard</span></div>
          <div className="stat-divider" />
          <div className="stat-item"><span className="stat-value">0</span><span className="stat-label">ML deps</span></div>
        </div>
      </section>

      {/* ═══ What's inside — 3 short pills ═══ */}
      <section className="landing-pills">
        <div className="pill-row">
          <div className="pill-card">
            <span className="pill-emoji">🧮</span>
            <span className="pill-text">Real math — every dot product shown</span>
          </div>
          <div className="pill-card">
            <span className="pill-emoji">⚡</span>
            <span className="pill-text">C++ engine running in your browser</span>
          </div>
          <div className="pill-card">
            <span className="pill-emoji">🎯</span>
            <span className="pill-text">Type your own prompts, see results</span>
          </div>
        </div>
      </section>

      {/* ═══ Under the Hood — Tech & Reasoning ═══ */}
      <section className="landing-under-hood">
        <div className="uh-header">
          <span className="uh-header-icon">🔧</span>
          <div>
            <h3 className="uh-title">Under the Hood</h3>
            <p className="uh-desc">What powers each module and what's real vs simulated</p>
          </div>
        </div>

        <div className="uh-grid">
          {/* Module cards */}
          <div className="uh-card">
            <div className="uh-card-header" style={{ borderColor: "#fbbf24" }}>
              <span className="uh-card-icon" style={{ color: "#fbbf24" }}>🧮</span>
              <span className="uh-card-name">Math Lab</span>
            </div>
            <div className="uh-card-body">
              <div className="uh-fact"><span className="uh-tag real">Real</span> C++ MatMul, Softmax, ReLU, GELU via WebAssembly</div>
              <div className="uh-fact"><span className="uh-tag real">Real</span> Step-by-step computation traces from engine</div>
              <div className="uh-fact"><span className="uh-tag sim">Demo</span> Uses small matrices (2x2, 3x3) for clarity</div>
            </div>
          </div>

          <div className="uh-card">
            <div className="uh-card-header" style={{ borderColor: "#4a90e2" }}>
              <span className="uh-card-icon" style={{ color: "#4a90e2" }}>💬</span>
              <span className="uh-card-name">Text AI Pipeline</span>
            </div>
            <div className="uh-card-body">
              <div className="uh-fact"><span className="uh-tag real">Real</span> BPE tokenizer (C++ vocabulary lookup)</div>
              <div className="uh-fact"><span className="uh-tag real">Real</span> Attention dot-product + softmax math</div>
              <div className="uh-fact"><span className="uh-tag sim">Demo</span> Embeddings are pseudo-random (not trained)</div>
              <div className="uh-fact"><span className="uh-tag sim">Demo</span> Predictions simulated (no real LLM weights)</div>
            </div>
          </div>

          <div className="uh-card">
            <div className="uh-card-header" style={{ borderColor: "#34d399" }}>
              <span className="uh-card-icon" style={{ color: "#34d399" }}>👁</span>
              <span className="uh-card-name">Vision AI</span>
            </div>
            <div className="uh-card-body">
              <div className="uh-fact"><span className="uh-tag real">Real</span> Convolution, ReLU, MaxPool computations</div>
              <div className="uh-fact"><span className="uh-tag real">Real</span> Trained MNIST model (98.5% accuracy)</div>
              <div className="uh-fact"><span className="uh-tag sim">Limited</span> Only digits 0-9 (MNIST dataset)</div>
            </div>
          </div>

          <div className="uh-card">
            <div className="uh-card-header" style={{ borderColor: "#a78bfa" }}>
              <span className="uh-card-icon" style={{ color: "#a78bfa" }}>✏️</span>
              <span className="uh-card-name">MNIST Live</span>
            </div>
            <div className="uh-card-body">
              <div className="uh-fact"><span className="uh-tag real">Real</span> Trained neural network (784→128→10)</div>
              <div className="uh-fact"><span className="uh-tag real">Real</span> Live inference via C++/WASM</div>
              <div className="uh-fact"><span className="uh-tag real">Real</span> Neuron activations visualized from actual forward pass</div>
            </div>
          </div>
        </div>

        {/* Tech stack strip */}
        <div className="uh-stack">
          <div className="uh-stack-title">Tech Stack</div>
          <div className="uh-stack-items">
            <div className="uh-stack-item">
              <span className="uh-stack-label">Engine</span>
              <span className="uh-stack-value">C++20 → WebAssembly</span>
            </div>
            <div className="uh-stack-item">
              <span className="uh-stack-label">Frontend</span>
              <span className="uh-stack-value">React 19 + Ant Design</span>
            </div>
            <div className="uh-stack-item">
              <span className="uh-stack-label">Build</span>
              <span className="uh-stack-value">Vite + Emscripten</span>
            </div>
            <div className="uh-stack-item">
              <span className="uh-stack-label">ML Deps</span>
              <span className="uh-stack-value">Zero (all from scratch)</span>
            </div>
            <div className="uh-stack-item">
              <span className="uh-stack-label">Dataset</span>
              <span className="uh-stack-value">MNIST (60K images)</span>
            </div>
            <div className="uh-stack-item">
              <span className="uh-stack-label">Compute</span>
              <span className="uh-stack-value">100% client-side</span>
            </div>
          </div>
        </div>

        {/* Why built this way */}
        <div className="uh-why">
          <div className="uh-why-title">Why build an AI engine from scratch?</div>
          <div className="uh-why-points">
            <div className="uh-why-point">
              <span className="uh-why-num">1</span>
              <span>To show that AI isn't magic — it's math you can trace step by step</span>
            </div>
            <div className="uh-why-point">
              <span className="uh-why-num">2</span>
              <span>To make every operation visible: attention, softmax, matmul — all computed live</span>
            </div>
            <div className="uh-why-point">
              <span className="uh-why-num">3</span>
              <span>To demonstrate systems engineering: SIMD kernels, memory allocators, graph compilation</span>
            </div>
          </div>
        </div>
      </section>

      {/* ═══ Footer ═══ */}
      <div className="landing-footer">
        <Space split={<Divider type="vertical" />} size="small" wrap>
          <Text type="secondary">Open source</Text>
          <Text type="secondary">MIT license</Text>
          <Text type="secondary">Apple Silicon optimized</Text>
        </Space>
      </div>
    </div>
  );
}
