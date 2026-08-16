import { useState, useEffect, useRef, useMemo } from "react";
import { Card, Button, Typography, Tag, Space, Radio, Alert, Progress, Tabs, Row, Col, Input } from "antd";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  TrophyOutlined,
  ArrowRightOutlined,
  ArrowLeftOutlined,
  ReloadOutlined,
  BulbOutlined,
  ExperimentOutlined,
  BookOutlined,
  CheckOutlined,
  CodeOutlined,
  RobotOutlined,
  PlayCircleOutlined,
  EyeOutlined,
} from "@ant-design/icons";
import {
  traceTokenize,
  traceEmbedding,
  traceAttention,
  traceMatMul,
  traceSoftmax,
  traceGELU,
} from "../engine/nn_wasm";
import StepViewer from "./StepViewer";
import "./TextAIJourney.css";

const { Title, Paragraph, Text } = Typography;

// 8 stages — complete end-to-end flow

const STAGES = [
  {
    id: "tokenize", title: "Tokenization", subtitle: "Your prompt → numbers",
    icon: "✂", color: "#fbbf24",
    visual: {
      description: "Type any text and see it split into tokens with their IDs:",
      input: "How does ChatGPT work?",
    },
    learn: {
      what: `When you type a message into ChatGPT, the very first thing that happens is tokenization. AI can't read text — it only understands numbers. A tokenizer splits your text into pieces (tokens) and assigns each a unique integer ID from a vocabulary of ~100K entries.`,
      why: "Without this, the neural network has nothing to work with. Every LLM starts here.",
      example: '"Hello world" → ["Hello", " world"] → [15496, 995]',
    },
    code: `// From inferx/nn/tokenizer.h
class Tokenizer {
  std::vector<int> encode(const std::string& text) {
    auto words = split_whitespace(text);
    for (const auto& word : words) {
      auto tokens = bpe_encode_word(word);
      ids.insert(ids.end(), tokens.begin(), tokens.end());
    }
    return ids; // [464, 3797, 3332, ...]
  }
};`,
    quiz: { question: "What does tokenization convert?", options: ["Numbers to text", "Text to numbers (integer IDs)", "Images to text", "Numbers to images"], correct: 1, explanation: "Tokenization converts raw text into integer IDs that neural networks can process." },
  },
  {
    id: "embed", title: "Embedding", subtitle: "Numbers → meaning vectors",
    icon: "📐", color: "#4a90e2",
    visual: {
      description: "Each token ID looks up a vector from a learned table. Similar meanings → nearby vectors:",
      input: null,
    },
    learn: {
      what: `Token IDs are arbitrary numbers. The embedding layer converts each ID into a rich vector (4096 numbers) that captures semantic meaning. "King" and "Queen" get similar vectors because they share meaning.`,
      why: 'This is how AI understands that words relate to each other — through the geometry of vector space.',
      example: 'token_id=15496 → embedding[15496] = [0.023, -0.041, 0.087, ...] (4096 dims)',
    },
    code: `// From inferx/nn/embedding.h
class Embedding {
  std::vector<float> forward(const std::vector<int>& token_ids) {
    for (int i = 0; i < seq_len; i++) {
      int id = token_ids[i];
      for (int j = 0; j < embed_dim_; j++)
        output[i * embed_dim_ + j] = weights_[id * embed_dim_ + j];
    }
    // Add positional encoding: sin/cos so model knows word ORDER
    return output; // [seq_len × embed_dim]
  }
};`,
    quiz: { question: 'Why do "King" and "Queen" have similar embedding vectors?', options: ["Similar spelling", "They share semantic meaning (both royalty)", "Same length", "Random"], correct: 1, explanation: "Words used in similar contexts end up with nearby vectors in embedding space." },
  },
  {
    id: "attention", title: "Self-Attention", subtitle: "Tokens understand context",
    icon: "◎", color: "#22d3ee",
    visual: {
      description: "Each token looks at every other token. Brighter = higher attention score:",
      input: null,
    },
    learn: {
      what: `THE breakthrough of transformers. Each token computes relevance scores with all other tokens, creating context-aware representations. "Bank" knows if you mean river bank or money bank by attending to surrounding words.`,
      why: 'Without attention, each token is processed independently — no context understanding.',
      example: '"The cat sat" → "sat" attends strongly to "cat" (subject-verb)',
    },
    code: `// From inferx/nn/attention.h
class Attention {
  std::vector<float> forward(const std::vector<float>& input, int seq_len) {
    auto Q = matmul(input, wq_, seq_len, d, d); // Query
    auto K = matmul(input, wk_, seq_len, d, d); // Key
    auto V = matmul(input, wv_, seq_len, d, d); // Value
    // scores[i][j] = (Q[i] · K[j]) / √d_k
    // weights = softmax(scores)
    // output[i] = Σ_j weights[i][j] × V[j]
    return output;
  }
};`,
    quiz: { question: 'In "The bank by the river", what helps the model know "bank" means river bank?', options: ["Tokenization", "Embedding alone", "Attention — 'bank' attends to 'river' with high score", "Random weights"], correct: 2, explanation: "Attention lets 'bank' look at all other tokens and get a high score with 'river'." },
  },
  {
    id: "matmul", title: "Matrix Multiply", subtitle: "The core computation",
    icon: "×", color: "#a78bfa",
    visual: {
      description: "Every layer is a matrix multiply. Watch each dot product compute:",
      input: null,
    },
    learn: {
      what: `Every linear layer, every attention projection, every FFN — all matrix multiplications. This single operation is ~90% of an LLM's compute time. When people say "AI needs GPUs", this is why.`,
      why: "It's how the network combines inputs with learned weights. The fundamental building block.",
      example: "C[i][j] = Σ A[i][k] × B[k][j] — one dot product per output cell",
    },
    code: `// From inferx/nn/matmul.h
class MatMul {
  static std::vector<float> forward(
      const std::vector<float>& A, int M, int K,
      const std::vector<float>& B, int K2, int N) {
    std::vector<float> C(M * N, 0.0f);
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++)
          C[i*N + j] += A[i*K + k] * B[k*N + j];
    return C; // Total FLOPs: 2×M×N×K
  }
};`,
    quiz: { question: "If A is [2×3] and B is [3×4], what shape is A×B?", options: ["[2×4]", "[3×3]", "[2×3]", "Can't multiply"], correct: 0, explanation: "A[M×K] × B[K×N] = C[M×N]. Here M=2, K=3, N=4 → result [2×4]." },
  },
  {
    id: "activation", title: "Activation (GELU)", subtitle: "Non-linearity",
    icon: "⌐", color: "#34d399",
    visual: {
      description: "Apply GELU to each value. Without this, 96 layers = 1 layer:",
      input: null,
    },
    learn: {
      what: `After each matrix multiply, GELU activation is applied. Without activation, stacking 96 layers collapses to just ONE linear layer — useless! GELU is smooth (unlike ReLU's hard cutoff), used in GPT/Claude.`,
      why: "Activation functions let the network learn complex, non-linear patterns.",
      example: "GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))",
    },
    code: `// From inferx/nn/activations.h
class Activations {
  static std::vector<float> gelu(const std::vector<float>& input) {
    const float c = sqrt(2.0f / M_PI);
    for (int i = 0; i < n; i++) {
      float x = input[i];
      float inner = c * (x + 0.044715f * x*x*x);
      output[i] = 0.5f * x * (1.0f + tanh(inner));
    }
    return output;
  }
};`,
    quiz: { question: "Why do neural networks NEED activation functions?", options: ["Speed", "Without them, 96 layers = 1 layer (no non-linearity)", "Memory savings", "They're optional"], correct: 1, explanation: "Without non-linearity, stacking layers is mathematically equivalent to one layer." },
  },
  {
    id: "transformer", title: "Transformer Stack", subtitle: "Repeat 96× (the LLM)",
    icon: "🏗", color: "#fb923c",
    visual: {
      description: "Stack attention + FFN into blocks. GPT-4 has 120 of these:",
      input: null,
    },
    learn: {
      what: `Everything above forms ONE transformer block. ChatGPT stacks 96-120 of these! Early layers capture syntax, middle layers semantics, late layers prepare for output. GPT-4: ~1.8T params across 120 layers.`,
      why: "This is what makes it 'Large'. Same architecture as your demo, wildly different scale.",
      example: "Block = LayerNorm → Attention → LayerNorm → FFN(MatMul→GELU→MatMul) × 96",
    },
    code: `// One transformer block (conceptual)
struct TransformerBlock {
  LayerNorm norm1, norm2;
  Attention  attn;
  Linear     ff_up;   // 4096 → 16384
  Activation gelu;
  Linear     ff_down; // 16384 → 4096

  Tensor forward(Tensor x) {
    x = x + attn(norm1(x));         // attention + residual
    x = x + ff_down(gelu(ff_up(norm2(x)))); // FFN + residual
    return x;
  }
};
// GPT-4: 120 blocks. Claude: 96+. Llama 70B: 80.`,
    quiz: { question: "What makes GPT-4 different from your 1-layer demo?", options: ["Different operations", "120 blocks with 1.8T params vs 1 block with ~50 params", "Different language", "It uses images"], correct: 1, explanation: "Same architecture, same operations — just massively more layers and parameters." },
  },
  {
    id: "generation", title: "Token Generation", subtitle: "Output → next word",
    icon: "🎯", color: "#f43f5e",
    visual: {
      description: "Softmax converts logits to probabilities. Highest wins:",
      input: null,
    },
    learn: {
      what: `After all transformer blocks, a final matmul projects back to vocab size (100K). Softmax converts to probabilities. The model picks the next token, appends it, and repeats. A 100-word response = ~130 full passes through 96 layers.`,
      why: "This is how ChatGPT writes: one token at a time, autoregressively.",
      example: '"The capital of France is" → P("Paris")=92%, P("the")=3%',
    },
    code: `// From inferx/nn/softmax.h
class Softmax {
  static std::vector<float> forward(const std::vector<float>& logits) {
    float max_val = *max_element(logits.begin(), logits.end());
    for (int i = 0; i < n; i++)
      exps[i] = exp(logits[i] - max_val);
    float sum = accumulate(exps.begin(), exps.end(), 0.0f);
    for (int i = 0; i < n; i++)
      probs[i] = exps[i] / sum;
    return probs; // sums to 1.0
  }
};`,
    quiz: { question: "How does ChatGPT generate a full paragraph?", options: ["All at once", "Predict one token, append, repeat through all layers", "From a database", "Random"], correct: 1, explanation: "LLMs are autoregressive — one token at a time. Each requires a full forward pass." },
  },
  {
    id: "rag", title: "RAG", subtitle: "When AI needs your docs",
    icon: "🔍", color: "#06b6d4",
    visual: {
      description: "Embed query → search vectors → augment prompt → generate answer:",
      input: null,
    },
    learn: {
      what: `LLMs only know training data. RAG (Retrieval-Augmented Generation) searches a vector database for relevant documents BEFORE generating, stuffing them into the context window. This is how enterprise chatbots work.`,
      why: "Without RAG, LLMs hallucinate about your docs. With RAG, they answer accurately.",
      example: 'Query → embed → search → "Context: {docs}\\nAnswer: {query}" → generate',
    },
    code: `// RAG Pipeline
struct RAGPipeline {
  Embedding  encoder;
  VectorDB   store;
  LLM        generator;

  std::string answer(const std::string& q) {
    auto vec = encoder.encode(q);       // 1. Embed
    auto docs = store.search(vec, 3);   // 2. Search
    std::string prompt = "Context:\\n";
    for (auto& d : docs) prompt += d.content + "\\n";
    prompt += "\\nQ: " + q + "\\nA:";  // 3. Augment
    return generator.generate(prompt);  // 4. Generate
  }
};`,
    quiz: { question: "What problem does RAG solve?", options: ["Speed", "Gives AI access to knowledge it wasn't trained on", "Reduces size", "Creativity"], correct: 1, explanation: "RAG retrieves documents at query time so the LLM can answer from your actual data." },
  },
];

// Next-token prediction dataset (simulated)

const PREDICTION_DB = {
  "the": [{ token: "cat", prob: 0.15 }, { token: "model", prob: 0.12 }, { token: "AI", prob: 0.10 }, { token: "best", prob: 0.08 }, { token: "world", prob: 0.07 }],
  "the capital of": [{ token: "France", prob: 0.35 }, { token: "the", prob: 0.12 }, { token: "India", prob: 0.10 }, { token: "Japan", prob: 0.08 }],
  "the capital of france is": [{ token: "Paris", prob: 0.92 }, { token: "a", prob: 0.03 }, { token: "the", prob: 0.02 }],
  "how does": [{ token: "AI", prob: 0.20 }, { token: "ChatGPT", prob: 0.18 }, { token: "this", prob: 0.12 }, { token: "it", prob: 0.10 }],
  "how does chatgpt": [{ token: "work", prob: 0.65 }, { token: "generate", prob: 0.12 }, { token: "know", prob: 0.08 }],
  "how does ai": [{ token: "work", prob: 0.45 }, { token: "learn", prob: 0.20 }, { token: "think", prob: 0.10 }],
  "what is": [{ token: "the", prob: 0.18 }, { token: "a", prob: 0.14 }, { token: "AI", prob: 0.10 }, { token: "machine", prob: 0.08 }, { token: "deep", prob: 0.06 }],
  "what is machine": [{ token: "learning", prob: 0.85 }, { token: "intelligence", prob: 0.05 }],
  "i want to": [{ token: "learn", prob: 0.25 }, { token: "build", prob: 0.18 }, { token: "understand", prob: 0.12 }, { token: "know", prob: 0.10 }],
  "i want to learn": [{ token: "about", prob: 0.30 }, { token: "AI", prob: 0.20 }, { token: "how", prob: 0.15 }, { token: "more", prob: 0.12 }],
  "hello": [{ token: "world", prob: 0.25 }, { token: "!", prob: 0.20 }, { token: "there", prob: 0.15 }, { token: ",", prob: 0.12 }],
  "hello world": [{ token: "!", prob: 0.30 }, { token: ".", prob: 0.20 }, { token: "how", prob: 0.10 }],
  "explain": [{ token: "how", prob: 0.25 }, { token: "the", prob: 0.15 }, { token: "what", prob: 0.12 }, { token: "this", prob: 0.10 }],
  "explain how": [{ token: "AI", prob: 0.20 }, { token: "neural", prob: 0.15 }, { token: "transformers", prob: 0.12 }, { token: "attention", prob: 0.10 }],
  "neural": [{ token: "network", prob: 0.70 }, { token: "networks", prob: 0.15 }, { token: "net", prob: 0.05 }],
  "neural network": [{ token: "is", prob: 0.20 }, { token: "can", prob: 0.12 }, { token: "works", prob: 0.10 }],
};

function getPredictions(text) {
  const lower = text.toLowerCase().trim();
  // Try exact match first, then last 3 words, then last 2, then last 1
  if (PREDICTION_DB[lower]) return PREDICTION_DB[lower];
  const words = lower.split(/\s+/);
  for (let n = Math.min(3, words.length); n >= 1; n--) {
    const key = words.slice(-n).join(" ");
    if (PREDICTION_DB[key]) return PREDICTION_DB[key];
  }
  // Fallback: generic predictions
  return [
    { token: "the", prob: 0.12 }, { token: "is", prob: 0.10 },
    { token: "a", prob: 0.08 }, { token: "and", prob: 0.07 },
    { token: "to", prob: 0.06 }, { token: "of", prob: 0.05 },
  ];
}

// Simple tokenizer for user input
function tokenizeInput(text) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  return words.map(w => ({
    text: w,
    id: Math.abs(w.split("").reduce((h, c) => ((h << 5) + h + c.charCodeAt(0)) | 0, 5381)) % 49900 + 100,
  }));
}

// Simple embedding visualization (generate pseudo-vectors)
function embedTokens(tokens) {
  return tokens.map(t => {
    const seed = t.id;
    const vec = [];
    let s = seed;
    for (let i = 0; i < 8; i++) {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      vec.push(((s >>> 16) / 65536) * 2 - 1);
    }
    return { ...t, embedding: vec };
  });
}

// Simple attention scores
function computeAttention(tokens) {
  const n = tokens.length;
  const scores = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) {
      const dist = Math.abs(i - j);
      const sim = (tokens[i].id * tokens[j].id) % 100 / 200;
      row.push(Math.exp(-dist * 0.4) * 0.6 + sim + 0.1);
    }
    const max = Math.max(...row);
    const exps = row.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    scores.push(exps.map(v => v / sum));
  }
  return scores;
}

// Main component

export default function TextAIJourney({ onComplete }) {
  const [userInput, setUserInput] = useState("How does ChatGPT work");
  const [currentStage, setCurrentStage] = useState(0);
  const [completedStages, setCompletedStages] = useState(() => {
    try { const s = localStorage.getItem("inferx-textai-completed"); return s ? JSON.parse(s) : []; } catch { return []; }
  });
  const [quizAnswer, setQuizAnswer] = useState(null);
  const [quizSubmitted, setQuizSubmitted] = useState(false);
  const [traceSteps, setTraceSteps] = useState([]);
  const [traceCurrentStep, setTraceCurrentStep] = useState(0);
  const contentRef = useRef(null);

  // Computed data from user input
  const tokens = useMemo(() => tokenizeInput(userInput), [userInput]);
  const embedded = useMemo(() => embedTokens(tokens), [tokens]);
  const attentionScores = useMemo(() => computeAttention(tokens), [tokens]);
  const predictions = useMemo(() => getPredictions(userInput), [userInput]);

  const stage = STAGES[currentStage];
  const progress = Math.round((completedStages.length / STAGES.length) * 100);

  useEffect(() => {
    try { localStorage.setItem("inferx-textai-completed", JSON.stringify(completedStages)); } catch {}
  }, [completedStages]);

  useEffect(() => {
    if (contentRef.current) contentRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [currentStage]);

  function goToStage(idx) {
    setCurrentStage(idx);
    setQuizAnswer(null);
    setQuizSubmitted(false);
    setTraceSteps([]);
  }

  function nextStage() { if (currentStage < STAGES.length - 1) goToStage(currentStage + 1); }
  function prevStage() { if (currentStage > 0) goToStage(currentStage - 1); }

  function submitQuiz() {
    setQuizSubmitted(true);
    if (quizAnswer === stage.quiz.correct && !completedStages.includes(stage.id)) {
      const updated = [...completedStages, stage.id];
      setCompletedStages(updated);
      if (updated.length === STAGES.length && onComplete) onComplete();
    }
  }

  function runExperiment() {
    let steps = [];
    switch (stage.id) {
      case "tokenize": steps = traceTokenize("Hello world how AI works"); break;
      case "embed": steps = traceEmbedding([15496, 995, 703, 9552, 2499]); break;
      case "attention": steps = traceAttention(5); break;
      case "matmul": steps = traceMatMul([1,2,3,4,5,6], 2, 3, [7,8,9,10,11,12], 3, 2); break;
      case "activation": steps = traceGELU([-2, -1, -0.5, 0, 0.5, 1, 2]); break;
      case "transformer": steps = traceAttention(4); break;
      case "generation": steps = traceSoftmax([2.5, 1.2, 0.8, -0.3, 3.1, 0.1]); break;
      case "rag": steps = traceEmbedding([464, 3797, 3332]); break;
    }
    setTraceSteps(steps || []);
    setTraceCurrentStep(0);
  }

  // Build tab items for the unified stage view
  const tabItems = [
    {
      key: "visual",
      label: <span><EyeOutlined /> Visual</span>,
      children: (
        <div className="taj-tab-content">
          {/* Show live results from user input */}
          {stage.id === "tokenize" && (
            <div className="taj-visual-result">
              <div className="taj-visual-label">Your input tokenized:</div>
              <div className="taj-token-chips">
                {tokens.map((t, i) => (
                  <div key={i} className="taj-token-chip">
                    <span className="taj-token-text">"{t.text}"</span>
                    <span className="taj-token-id">ID: {t.id}</span>
                  </div>
                ))}
              </div>
              <div className="taj-visual-summary">
                <Tag>{tokens.length} tokens</Tag>
                <Tag color="blue">IDs: [{tokens.map(t => t.id).join(", ")}]</Tag>
              </div>
            </div>
          )}

          {stage.id === "embed" && (
            <div className="taj-visual-result">
              <div className="taj-visual-label">Each token → vector (showing first 8 of 4096 dims):</div>
              <div className="taj-embed-grid">
                {embedded.slice(0, 6).map((t, i) => (
                  <div key={i} className="taj-embed-row">
                    <span className="taj-embed-token">"{t.text}"</span>
                    <div className="taj-embed-bars">
                      {t.embedding.map((v, j) => (
                        <div key={j} className="taj-embed-bar" style={{ height: `${Math.abs(v) * 24}px`, background: v > 0 ? "#4a90e2" : "#f87171", opacity: 0.4 + Math.abs(v) * 0.6 }} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {stage.id === "attention" && (
            <div className="taj-visual-result">
              <div className="taj-visual-label">Attention heatmap — how much each token looks at others:</div>
              <div className="taj-attn-grid">
                <div className="taj-attn-header">
                  <div className="taj-attn-corner"></div>
                  {tokens.slice(0, 6).map((t, j) => <div key={j} className="taj-attn-col-label">{t.text}</div>)}
                </div>
                {attentionScores.slice(0, 6).map((row, i) => (
                  <div key={i} className="taj-attn-row">
                    <div className="taj-attn-row-label">{tokens[i]?.text}</div>
                    {row.slice(0, 6).map((score, j) => (
                      <div key={j} className="taj-attn-cell" style={{ background: `rgba(74, 144, 226, ${score})`, color: score > 0.3 ? "#fff" : "#94a3b8" }}>
                        {(score * 100).toFixed(0)}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}

          {stage.id === "matmul" && (
            <div className="taj-visual-result">
              <div className="taj-visual-label">Matrix multiply: input × weights = output (shown for first token)</div>
              <div className="taj-matmul-visual">
                <div className="taj-matmul-eq">
                  <div className="taj-mat">[{embedded[0]?.embedding.slice(0, 4).map(v => v.toFixed(2)).join(", ")}]</div>
                  <span className="taj-matmul-op">×</span>
                  <div className="taj-mat">[W<sub>4096×4096</sub>]</div>
                  <span className="taj-matmul-op">=</span>
                  <div className="taj-mat">[{embedded[0]?.embedding.slice(0, 4).map(v => (v * 1.3 + 0.1).toFixed(2)).join(", ")}]</div>
                </div>
                <Text type="secondary" style={{ marginTop: 8, display: "block" }}>Each token's vector is multiplied by learned weight matrices. {tokens.length} tokens × 4096 dims = {tokens.length * 4096} multiply-accumulate operations per layer.</Text>
              </div>
            </div>
          )}

          {stage.id === "activation" && (
            <div className="taj-visual-result">
              <div className="taj-visual-label">GELU applied to each value (negative values reduced, positive kept):</div>
              <div className="taj-activation-visual">
                {embedded[0]?.embedding.map((v, i) => {
                  const gelu = 0.5 * v * (1 + Math.tanh(Math.sqrt(2/Math.PI) * (v + 0.044715 * v*v*v)));
                  return (
                    <div key={i} className="taj-act-item">
                      <span className="taj-act-in" style={{ color: v < 0 ? "#f87171" : "#4a90e2" }}>{v.toFixed(3)}</span>
                      <span className="taj-act-arrow">→</span>
                      <span className="taj-act-out" style={{ color: gelu < 0.01 ? "#64748b" : "#34d399" }}>{gelu.toFixed(3)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {stage.id === "transformer" && (
            <div className="taj-visual-result">
              <div className="taj-visual-label">Your {tokens.length} tokens pass through 96 transformer blocks:</div>
              <div className="taj-transformer-visual">
                {[1, 2, 3, "...", 94, 95, 96].map((block, i) => (
                  <div key={i} className={`taj-block ${block === "..." ? "dots" : ""}`}>
                    {block === "..." ? "⋮" : `Block ${block}`}
                  </div>
                ))}
              </div>
              <Text type="secondary">Each block: LayerNorm → Attention → LayerNorm → FFN. Total: {tokens.length * 96} attention computations.</Text>
            </div>
          )}

          {stage.id === "generation" && (
            <div className="taj-visual-result">
              <div className="taj-visual-label">Next token prediction for your input:</div>
              <div className="taj-predictions">
                {predictions.map((p, i) => (
                  <div key={i} className={`taj-pred-item ${i === 0 ? "winner" : ""}`}>
                    <span className="taj-pred-token">"{p.token}"</span>
                    <div className="taj-pred-bar">
                      <div className="taj-pred-fill" style={{ width: `${p.prob * 100}%`, background: i === 0 ? "#34d399" : "#4a90e2" }} />
                    </div>
                    <span className="taj-pred-prob">{(p.prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              <div className="taj-pred-result">
                <Text>Predicted next word: </Text>
                <Tag color="green" style={{ fontSize: 16, padding: "4px 12px" }}>"{predictions[0]?.token}"</Tag>
                <Text type="secondary"> with {(predictions[0]?.prob * 100).toFixed(1)}% confidence</Text>
              </div>
            </div>
          )}

          {stage.id === "rag" && (
            <div className="taj-visual-result">
              <div className="taj-visual-label">If your query needs external knowledge, RAG retrieves docs first:</div>
              <div className="taj-rag-visual">
                <div className="taj-rag-step"><Tag color="blue">1</Tag> Embed: "{userInput}" → vector</div>
                <div className="taj-rag-step"><Tag color="purple">2</Tag> Search: find similar docs in vector DB</div>
                <div className="taj-rag-step"><Tag color="orange">3</Tag> Augment: stuff docs into prompt context</div>
                <div className="taj-rag-step"><Tag color="green">4</Tag> Generate: LLM answers using the docs</div>
              </div>
            </div>
          )}

          {/* Also allow running the C++ engine trace */}
          <div style={{ marginTop: 16, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
            <Button size="small" icon={<PlayCircleOutlined />} onClick={runExperiment}>
              Run C++ Engine Trace
            </Button>
            {traceSteps.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <StepViewer steps={traceSteps} currentStep={traceCurrentStep} onStepChange={setTraceCurrentStep} title={stage.title} />
              </div>
            )}
          </div>
        </div>
      ),
    },
    {
      key: "learn",
      label: <span><BookOutlined /> Learn</span>,
      children: (
        <div className="taj-tab-content">
          <Paragraph style={{ fontSize: 15, lineHeight: 1.8 }}>{stage.learn.what}</Paragraph>
          <Alert type="info" showIcon icon={<BulbOutlined />} message={stage.learn.why} style={{ marginBottom: 16 }} />
          <div className="taj-example-code"><code>{stage.learn.example}</code></div>
        </div>
      ),
    },
    {
      key: "code",
      label: <span><CodeOutlined /> C++ Code</span>,
      children: (
        <div className="taj-tab-content">
          <Space style={{ marginBottom: 12 }}>
            <Tag color="blue">Real InferX engine code</Tag>
            <Text type="secondary" style={{ fontSize: 12 }}>Runs in your browser via WebAssembly</Text>
          </Space>
          <div className="taj-cpp-code"><pre><code>{stage.code}</code></pre></div>
        </div>
      ),
    },
    {
      key: "quiz",
      label: <span><CheckOutlined /> Quiz {completedStages.includes(stage.id) && <CheckCircleOutlined style={{ color: "#34d399", marginLeft: 4 }} />}</span>,
      children: (
        <div className="taj-tab-content">
          <Paragraph strong style={{ fontSize: 15 }}>{stage.quiz.question}</Paragraph>
          <Radio.Group
            value={quizAnswer}
            onChange={(e) => setQuizAnswer(e.target.value)}
            disabled={quizSubmitted}
            className="taj-quiz-options"
          >
            <Space direction="vertical" style={{ width: "100%" }}>
              {stage.quiz.options.map((opt, i) => (
                <Radio key={i} value={i}>{opt}</Radio>
              ))}
            </Space>
          </Radio.Group>

          {!quizSubmitted && (
            <Button type="primary" onClick={submitQuiz} disabled={quizAnswer === null} style={{ marginTop: 16 }} icon={<CheckOutlined />}>
              Submit
            </Button>
          )}

          {quizSubmitted && (
            <div style={{ marginTop: 12 }}>
              {quizAnswer === stage.quiz.correct ? (
                <Alert type="success" showIcon message="Correct!" description={stage.quiz.explanation} />
              ) : (
                <Alert type="error" showIcon message={`Incorrect — answer: ${stage.quiz.options[stage.quiz.correct]}`} description={stage.quiz.explanation} />
              )}
              <div style={{ marginTop: 12 }}>
                {quizAnswer === stage.quiz.correct && currentStage < STAGES.length - 1 && (
                  <Button type="primary" icon={<ArrowRightOutlined />} onClick={nextStage}>Next: {STAGES[currentStage + 1].title}</Button>
                )}
                {quizAnswer !== stage.quiz.correct && (
                  <Space>
                    <Button icon={<ReloadOutlined />} onClick={() => { setQuizAnswer(null); setQuizSubmitted(false); }}>Try Again</Button>
                  </Space>
                )}
                {quizAnswer === stage.quiz.correct && currentStage === STAGES.length - 1 && (
                  <Alert type="success" showIcon icon={<TrophyOutlined />} message="Complete! You understand the entire Text AI pipeline." />
                )}
              </div>
            </div>
          )}
        </div>
      ),
    },
  ];

  return (
    <div className="textai-journey" ref={contentRef}>
      {/* Header + Progress */}
      <div className="taj-header">
        <Title level={2}><RobotOutlined style={{ marginRight: 8 }} />How Text AI Works</Title>
        <Paragraph type="secondary">
          Complete end-to-end: your prompt flows through 8 stages to become a response. Each stage: watch it happen, learn why, read the C++ code, then prove you get it.
        </Paragraph>
        <Progress percent={progress} steps={8} strokeColor="#4a90e2" size="small" format={() => `${completedStages.length}/8`} />
      </div>

      {/* User input section */}
      <div className="taj-input-section">
        <div className="taj-input-label">
          <RobotOutlined style={{ marginRight: 6 }} />
          <Text strong>Your prompt (this will flow through all 8 stages):</Text>
        </div>
        <Input
          size="large"
          placeholder="Type anything... e.g. 'How does ChatGPT work'"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          className="taj-input-field"
          prefix={<span style={{ color: "#64748b" }}>💬</span>}
          allowClear
        />
        <div className="taj-input-preview">
          <Text type="secondary" style={{ fontSize: 12 }}>
            → {tokens.length} tokens → predicted next: "<Text strong style={{ color: "#34d399" }}>{predictions[0]?.token}</Text>" ({(predictions[0]?.prob * 100).toFixed(0)}%)
          </Text>
        </div>
      </div>

      {/* Stage Navigation — scrollable pill chips */}
      <div className="taj-stage-nav">
        {STAGES.map((s, i) => {
          const isActive = i === currentStage;
          const isDone = completedStages.includes(s.id);
          return (
            <button
              key={s.id}
              className={`taj-pill ${isActive ? "active" : ""} ${isDone ? "done" : ""}`}
              onClick={() => goToStage(i)}
              style={isActive ? { borderColor: s.color, background: `${s.color}15` } : {}}
            >
              <span className="taj-pill-icon">{isDone ? <CheckCircleOutlined style={{ color: "#34d399" }} /> : <span className="taj-pill-num">{i + 1}</span>}</span>
              <span className="taj-pill-label">{s.title}</span>
            </button>
          );
        })}
      </div>

      {/* Stage card */}
      <Card
        className="taj-stage-card"
        style={{ borderTop: `3px solid ${stage.color}` }}
        title={
          <Space>
            <span className="taj-stage-icon-inline" style={{ color: stage.color }}>{stage.icon}</span>
            <span>Stage {currentStage + 1}: {stage.title}</span>
            <Tag style={{ marginLeft: 8 }}>{stage.subtitle}</Tag>
          </Space>
        }
      >
        <Tabs items={tabItems} defaultActiveKey="visual" />
      </Card>

      {/* Bottom Navigation */}
      <div className="taj-bottom-nav">
        <Button disabled={currentStage === 0} icon={<ArrowLeftOutlined />} onClick={prevStage}>Previous</Button>
        <Text type="secondary">Stage {currentStage + 1} of {STAGES.length}</Text>
        <Button type="primary" disabled={currentStage === STAGES.length - 1} onClick={nextStage}>Next <ArrowRightOutlined /></Button>
      </div>
    </div>
  );
}
