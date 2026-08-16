import { useState, useEffect, useRef, useMemo } from "react";
import { Card, Button, Typography, Tag, Space, Radio, Alert, Steps, Result, Divider, Row, Col, Slider, Switch, Tooltip } from "antd";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  TrophyOutlined,
  ArrowRightOutlined,
  ReloadOutlined,
  BulbOutlined,
  ExperimentOutlined,
  BookOutlined,
  CheckOutlined,
  ThunderboltOutlined,
  RobotOutlined,
  MessageOutlined,
} from "@ant-design/icons";
import "./LLMExplorer.css";

const { Title, Paragraph, Text } = Typography;

// Lesson data

const LESSON_SECTIONS = [
  {
    title: "What is an LLM?",
    content: `When you type a message to ChatGPT or Claude, you're talking to a Large Language Model. 
    It's not magic — it's the exact same building blocks you already learned (tokenization, embeddings, attention, matmul, softmax) 
    stacked into a massive tower of 96+ transformer blocks with billions of parameters.`,
    highlight: "An LLM is just your Text Pipeline repeated 96 times with billions of learned weights.",
    analogy: {
      title: "Think of it like this:",
      text: `You learned to write one sentence in grade school. An LLM is like a student who read 
      every book ever written and can now write fluent paragraphs — using the same grammar rules you learned, 
      just applied with way more experience (training data).`,
    },
  },
  {
    title: "How does ChatGPT generate text?",
    content: `Here's the key insight: LLMs generate ONE TOKEN AT A TIME. When Claude writes a paragraph, 
    it's actually predicting the next word, then the next, then the next — hundreds of times in sequence.`,
    highlight: "Autoregressive = predict next token, append it, repeat.",
    steps: [
      'You type: "The capital of France is"',
      "LLM runs all 96 transformer blocks on your input",
      'Output layer produces probabilities: "Paris" (92%), "the" (3%), "Lyon" (1%)...',
      '"Paris" is selected and appended to the sequence',
      'Now input is: "The capital of France is Paris" → predict next token again',
      'Repeats until it generates a stop token or hits max length',
    ],
  },
  {
    title: "Inside the architecture",
    content: `GPT-4, Claude 3.5, Llama 3 — they all use the same fundamental architecture: the Transformer.`,
    architecture: {
      layers: [
        { name: "Token Embedding", desc: "Convert token IDs to 4096-dim vectors", size: "vocab × 4096" },
        { name: "Position Encoding", desc: "Add position information (where each token sits)", size: "context × 4096" },
        { name: "Transformer Block × 96", desc: "Each block: LayerNorm → Multi-Head Attention → LayerNorm → FFN", size: "Billions of params" },
        { name: "Output Projection", desc: "Project back to vocabulary size → logits for each possible next token", size: "4096 × vocab" },
        { name: "Softmax", desc: "Convert logits to probabilities", size: "vocab probs" },
      ],
      comparison: [
        { model: "GPT-4", params: "~1.8T (rumored)", layers: "120", context: "128K tokens" },
        { model: "Claude 3.5", params: "~175B (estimated)", layers: "96+", context: "200K tokens" },
        { model: "Llama 3 70B", params: "70B", layers: "80", context: "8K tokens" },
        { model: "Your Text Pipeline", params: "~50", layers: "1", context: "10 tokens" },
      ],
    },
  },
  {
    title: "Temperature & Sampling",
    content: `When the model outputs probabilities for the next token, how do we pick one? 
    This is where temperature and sampling strategies come in — and it's why the same prompt can give different answers each time.`,
    highlight: "Temperature controls creativity. Low = predictable. High = creative/random.",
    temperature: {
      low: { value: 0.1, behavior: "Almost always picks the highest probability token. Very predictable, repetitive.", example: '"The sky is blue. The sky is blue. The sky is..."' },
      medium: { value: 0.7, behavior: "Usually picks likely tokens but sometimes surprises. Good balance.", example: '"The sky is a canvas of shifting colors at dusk."' },
      high: { value: 1.5, behavior: "Often picks unlikely tokens. Creative but can be nonsensical.", example: '"The sky is a refrigerator of purple elephants dancing."' },
    },
  },
];

// Quiz data

const QUIZ = [
  {
    question: "How does ChatGPT generate a full paragraph of text?",
    options: [
      "It writes the whole paragraph at once in one computation",
      "It predicts one token at a time, appends it, and repeats",
      "It retrieves pre-written paragraphs from a database",
      "It generates all words in parallel simultaneously",
    ],
    correct: 1,
    explanation: "LLMs are autoregressive — they predict one token at a time, add it to the context, then predict the next. A 100-word response requires ~130 forward passes through the entire model.",
  },
  {
    question: "What makes an LLM 'large' compared to your Text Pipeline demo?",
    options: [
      "It uses different math operations",
      "It has billions of parameters and 96+ transformer layers (vs your 1 layer with ~50 params)",
      "It uses a completely different architecture",
      "It runs on special hardware only",
    ],
    correct: 1,
    explanation: "Same architecture (transformer), same operations (attention, matmul, softmax). The difference is scale: GPT-4 has ~1.8 trillion parameters across 120 layers. Your demo has 1 layer.",
  },
  {
    question: "If temperature = 0, what happens?",
    options: [
      "The model refuses to generate text",
      "It always picks the most probable next token (greedy, deterministic)",
      "It picks tokens completely at random",
      "It generates faster",
    ],
    correct: 1,
    explanation: "Temperature=0 means greedy decoding — always pick the highest probability token. Same prompt = same output every time. This is why ChatGPT can seem repetitive at low temps.",
  },
  {
    question: "Why can Claude handle 200K tokens of context but GPT-2 only handled 1024?",
    options: [
      "Claude uses a different programming language",
      "Advances in positional encoding (RoPE) and attention optimization (FlashAttention)",
      "Claude has more RAM",
      "GPT-2 was intentionally limited",
    ],
    correct: 1,
    explanation: "Long context requires: (1) positional encodings that generalize to unseen lengths (RoPE), (2) efficient attention that doesn't OOM on n² memory (FlashAttention), and (3) training on long documents.",
  },
  {
    question: "When you ask Claude 'explain quantum physics simply', what does the model actually output?",
    options: [
      "A pre-stored explanation from its training data",
      "Probability distributions over vocabulary tokens, sampled one at a time",
      "A search result from the internet",
      "A compressed version of Wikipedia",
    ],
    correct: 1,
    explanation: "The model outputs a probability distribution over its entire vocabulary (~100K tokens) for each position. It samples from this distribution, appends the token, and repeats. No memorized answers — it generates fresh each time.",
  },
];

// Token-by-token generation simulator

// Simulated vocabulary with probabilities for demo
const VOCAB_DEMO = {
  "The capital of France is": [
    { token: " Paris", prob: 0.92 },
    { token: " the", prob: 0.03 },
    { token: " a", prob: 0.02 },
    { token: " Lyon", prob: 0.01 },
    { token: " located", prob: 0.01 },
    { token: " not", prob: 0.005 },
    { token: " beautiful", prob: 0.005 },
  ],
  "The capital of France is Paris": [
    { token: ".", prob: 0.55 },
    { token: ",", prob: 0.25 },
    { token: " and", prob: 0.08 },
    { token: " which", prob: 0.05 },
    { token: "!", prob: 0.04 },
    { token: " -", prob: 0.02 },
    { token: " (", prob: 0.01 },
  ],
  "The capital of France is Paris.": [
    { token: " It", prob: 0.35 },
    { token: " Paris", prob: 0.15 },
    { token: " The", prob: 0.12 },
    { token: " Known", prob: 0.10 },
    { token: " This", prob: 0.08 },
    { token: "\n", prob: 0.10 },
    { token: " France", prob: 0.05 },
  ],
  "The capital of France is Paris. It": [
    { token: " is", prob: 0.45 },
    { token: " has", prob: 0.20 },
    { token: "'s", prob: 0.15 },
    { token: " was", prob: 0.08 },
    { token: " remains", prob: 0.05 },
    { token: " sits", prob: 0.04 },
    { token: " became", prob: 0.03 },
  ],
  "The capital of France is Paris. It is": [
    { token: " known", prob: 0.30 },
    { token: " the", prob: 0.18 },
    { token: " a", prob: 0.15 },
    { token: " home", prob: 0.12 },
    { token: " famous", prob: 0.10 },
    { token: " also", prob: 0.08 },
    { token: " one", prob: 0.07 },
  ],
  "The capital of France is Paris. It is known": [
    { token: " for", prob: 0.55 },
    { token: " as", prob: 0.30 },
    { token: " worldwide", prob: 0.05 },
    { token: " to", prob: 0.04 },
    { token: " globally", prob: 0.03 },
    { token: " around", prob: 0.02 },
    { token: " internationally", prob: 0.01 },
  ],
};

const PROMPTS = [
  "The capital of France is",
  "AI works by",
  "When you ask Claude",
];

// Simple hash-based deterministic "probabilities" for prompts we don't have pre-made data for
function generateFakeProbs(text) {
  const words = ["the", "is", "a", "an", "and", "of", "to", "in", "it", "for", "that", "was", "on", "with", "as", "are", "this", "by", "from", "be"];
  const hash = text.split("").reduce((a, c) => a + c.charCodeAt(0), 0);
  const shuffled = [...words].sort((a, b) => ((hash + a.charCodeAt(0)) % 7) - ((hash + b.charCodeAt(0)) % 7));
  const probs = [0.35, 0.20, 0.15, 0.10, 0.07, 0.05, 0.04, 0.02, 0.01, 0.01];
  return shuffled.slice(0, 7).map((w, i) => ({ token: ` ${w}`, prob: probs[i] || 0.01 }));
}

function getNextTokenProbs(context) {
  return VOCAB_DEMO[context] || generateFakeProbs(context);
}

function sampleToken(probs, temperature) {
  if (temperature <= 0.01) {
    return probs[0]; // greedy
  }
  // Apply temperature
  const scaled = probs.map(p => ({ ...p, prob: Math.pow(p.prob, 1 / temperature) }));
  const sum = scaled.reduce((a, p) => a + p.prob, 0);
  const normalized = scaled.map(p => ({ ...p, prob: p.prob / sum }));
  
  // Sample
  const r = Math.random();
  let cumulative = 0;
  for (const p of normalized) {
    cumulative += p.prob;
    if (r < cumulative) return p;
  }
  return normalized[normalized.length - 1];
}

// Main component

export default function LLMExplorer({ onComplete }) {
  const [mode, setMode] = useState("learn"); // "learn" | "experiment" | "quiz" | "result"
  const [lessonStep, setLessonStep] = useState(0);
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizSubmitted, setQuizSubmitted] = useState(false);
  const [showSolution, setShowSolution] = useState({});

  // Experiment state
  const [genContext, setGenContext] = useState("The capital of France is");
  const [genTokens, setGenTokens] = useState([]);
  const [temperature, setTemperature] = useState(0.7);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentProbs, setCurrentProbs] = useState(null);
  const [selectedToken, setSelectedToken] = useState(null);
  const genTimerRef = useRef(null);

  // Scroll to top on mode change
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [mode]);

  // Cleanup timer
  useEffect(() => () => { if (genTimerRef.current) clearInterval(genTimerRef.current); }, []);

  // --- Experiment functions
  function startGeneration() {
    setGenTokens([]);
    setSelectedToken(null);
    const probs = getNextTokenProbs(genContext);
    setCurrentProbs(probs);
  }

  function pickToken(token) {
    setSelectedToken(token);
    const newContext = genContext + token.token;
    const newTokens = [...genTokens, { token: token.token, prob: token.prob }];
    setGenTokens(newTokens);

    // Small delay then show next probs
    setTimeout(() => {
      setGenContext(newContext);
      setSelectedToken(null);
      if (newTokens.length < 6) {
        setCurrentProbs(getNextTokenProbs(newContext));
      } else {
        setCurrentProbs(null); // End generation
      }
    }, 500);
  }

  function autoGenerate() {
    setIsGenerating(true);
    setGenTokens([]);
    setSelectedToken(null);
    let ctx = genContext;
    let tokens = [];
    let step = 0;

    genTimerRef.current = setInterval(() => {
      const probs = getNextTokenProbs(ctx);
      setCurrentProbs(probs);
      const chosen = sampleToken(probs, temperature);
      setSelectedToken(chosen);

      tokens = [...tokens, { token: chosen.token, prob: chosen.prob }];
      setGenTokens([...tokens]);
      ctx = ctx + chosen.token;

      step++;
      if (step >= 6) {
        clearInterval(genTimerRef.current);
        setTimeout(() => {
          setGenContext(ctx);
          setCurrentProbs(null);
          setSelectedToken(null);
          setIsGenerating(false);
        }, 400);
      } else {
        setTimeout(() => {
          setGenContext(ctx);
          setSelectedToken(null);
          setCurrentProbs(getNextTokenProbs(ctx));
        }, 400);
      }
    }, 1200);
  }

  function resetExperiment() {
    if (genTimerRef.current) clearInterval(genTimerRef.current);
    setGenContext("The capital of France is");
    setGenTokens([]);
    setCurrentProbs(null);
    setSelectedToken(null);
    setIsGenerating(false);
  }

  // --- Quiz functions
  function submitQuiz() {
    setQuizSubmitted(true);
    const correct = QUIZ.filter((q, i) => quizAnswers[i] === q.correct).length;
    if (correct >= 3 && onComplete) {
      onComplete();
    }
    setMode("result");
  }

  function retryQuiz() {
    setQuizAnswers({});
    setQuizSubmitted(false);
    setShowSolution({});
    setMode("quiz");
  }

  const quizScore = useMemo(() => {
    if (!quizSubmitted) return 0;
    return QUIZ.filter((q, i) => quizAnswers[i] === q.correct).length;
  }, [quizSubmitted, quizAnswers]);

  const passed = quizScore >= 3;

  // Render
  // ═══════════════════════════════════════════════════════════════

  return (
    <div className="llm-explorer animate-in">
      {/* Header */}
      <div className="llm-header">
        <Title level={2} style={{ marginBottom: 4 }}>
          <RobotOutlined style={{ marginRight: 8 }} />
          How LLMs Work
        </Title>
        <Paragraph type="secondary">
          Understand what happens when you send a message to ChatGPT or Claude — from prompt to generated response.
        </Paragraph>
        <Space className="llm-mode-tabs">
          <Button type={mode === "learn" ? "primary" : "default"} icon={<BookOutlined />} onClick={() => setMode("learn")}>
            Learn
          </Button>
          <Button type={mode === "experiment" ? "primary" : "default"} icon={<ExperimentOutlined />} onClick={() => setMode("experiment")}>
            Experiment
          </Button>
          <Button type={mode === "quiz" || mode === "result" ? "primary" : "default"} icon={<CheckOutlined />} onClick={() => setMode("quiz")}>
            Quiz
          </Button>
        </Space>
      </div>

      {/* Learn mode */}
      {mode === "learn" && (
        <div className="llm-learn">
          {/* Lesson navigation */}
          <div className="lesson-nav">
            <Steps
              current={lessonStep}
              size="small"
              onChange={setLessonStep}
              items={LESSON_SECTIONS.map((s, i) => ({ title: s.title }))}
            />
          </div>

          {/* Current lesson */}
          <div className="lesson-content">
            <Card bordered={false} className="lesson-main-card">
              <Title level={4}>{LESSON_SECTIONS[lessonStep].title}</Title>
              <Paragraph style={{ fontSize: 15, lineHeight: 1.8 }}>
                {LESSON_SECTIONS[lessonStep].content}
              </Paragraph>

              {/* Highlight box */}
              {LESSON_SECTIONS[lessonStep].highlight && (
                <Alert
                  type="info"
                  showIcon
                  icon={<BulbOutlined />}
                  message={LESSON_SECTIONS[lessonStep].highlight}
                  className="lesson-highlight"
                />
              )}

              {/* Analogy */}
              {LESSON_SECTIONS[lessonStep].analogy && (
                <Card size="small" className="lesson-analogy" bordered={false}>
                  <Text strong>{LESSON_SECTIONS[lessonStep].analogy.title}</Text>
                  <Paragraph type="secondary" style={{ marginBottom: 0, marginTop: 4 }}>
                    {LESSON_SECTIONS[lessonStep].analogy.text}
                  </Paragraph>
                </Card>
              )}

              {/* Numbered steps */}
              {LESSON_SECTIONS[lessonStep].steps && (
                <div className="lesson-steps-visual">
                  {LESSON_SECTIONS[lessonStep].steps.map((step, i) => (
                    <div key={i} className="gen-step-item">
                      <span className="gen-step-num">{i + 1}</span>
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Architecture diagram */}
              {LESSON_SECTIONS[lessonStep].architecture && (
                <div className="arch-section">
                  <Title level={5}>The Transformer Stack</Title>
                  <div className="arch-layers">
                    {LESSON_SECTIONS[lessonStep].architecture.layers.map((layer, i) => (
                      <div key={i} className="arch-layer-card">
                        <div className="arch-layer-name">{layer.name}</div>
                        <div className="arch-layer-desc">{layer.desc}</div>
                        <Tag color="blue">{layer.size}</Tag>
                      </div>
                    ))}
                  </div>

                  <Divider />
                  <Title level={5}>Scale Comparison: Your Demo vs Real LLMs</Title>
                  <div className="model-comparison">
                    {LESSON_SECTIONS[lessonStep].architecture.comparison.map((m, i) => (
                      <div key={i} className={`model-card ${i === 3 ? "yours" : ""}`}>
                        <Text strong>{m.model}</Text>
                        <div className="model-stats">
                          <Tag>{m.params} params</Tag>
                          <Tag color="blue">{m.layers} layers</Tag>
                          <Tag color="green">{m.context} ctx</Tag>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Temperature visualization */}
              {LESSON_SECTIONS[lessonStep].temperature && (
                <div className="temp-section">
                  <Title level={5}>Temperature Examples</Title>
                  <Row gutter={[16, 16]}>
                    {Object.entries(LESSON_SECTIONS[lessonStep].temperature).map(([key, val]) => (
                      <Col xs={24} md={8} key={key}>
                        <Card size="small" className={`temp-card temp-${key}`} bordered={false}>
                          <Tag color={key === "low" ? "blue" : key === "medium" ? "green" : "red"}>
                            T = {val.value}
                          </Tag>
                          <Paragraph style={{ fontSize: 13, marginTop: 8 }}>{val.behavior}</Paragraph>
                          <code className="temp-example">{val.example}</code>
                        </Card>
                      </Col>
                    ))}
                  </Row>
                </div>
              )}
            </Card>

            {/* Navigation */}
            <div className="lesson-nav-btns">
              <Button
                disabled={lessonStep === 0}
                onClick={() => setLessonStep(lessonStep - 1)}
              >
                Previous
              </Button>
              {lessonStep < LESSON_SECTIONS.length - 1 ? (
                <Button type="primary" onClick={() => setLessonStep(lessonStep + 1)}>
                  Next <ArrowRightOutlined />
                </Button>
              ) : (
                <Button type="primary" icon={<ExperimentOutlined />} onClick={() => setMode("experiment")}>
                  Try It: Generate Tokens
                </Button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Experiment mode */}
      {mode === "experiment" && (
        <div className="llm-experiment">
          <Alert
            type="info"
            showIcon
            icon={<ExperimentOutlined />}
            message="Token-by-Token Generation Simulator"
            description="See exactly how ChatGPT/Claude generates text. Pick tokens manually or let the model auto-generate with temperature control."
            style={{ marginBottom: 20 }}
          />

          {/* Controls */}
          <Card bordered={false} className="exp-controls">
            <Row gutter={16} align="middle">
              <Col flex="auto">
                <Text strong>Temperature: </Text>
                <Slider
                  min={0}
                  max={2}
                  step={0.1}
                  value={temperature}
                  onChange={setTemperature}
                  style={{ width: 200, display: "inline-block", margin: "0 16px" }}
                  marks={{ 0: "Greedy", 0.7: "Default", 1.5: "Creative", 2: "Chaos" }}
                />
                <Tag color={temperature < 0.3 ? "blue" : temperature < 1 ? "green" : "red"}>
                  T = {temperature.toFixed(1)}
                </Tag>
              </Col>
              <Col>
                <Space>
                  <Button onClick={startGeneration} disabled={isGenerating}>
                    Manual Mode
                  </Button>
                  <Button type="primary" icon={<ThunderboltOutlined />} onClick={autoGenerate} disabled={isGenerating}>
                    Auto Generate
                  </Button>
                  <Button icon={<ReloadOutlined />} onClick={resetExperiment}>
                    Reset
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>

          {/* Generated text display */}
          <Card bordered={false} className="exp-output">
            <Text type="secondary" style={{ fontSize: 12 }}>Generated sequence:</Text>
            <div className="gen-text-display">
              <span className="gen-prompt">{genContext.split("").slice(0, "The capital of France is".length).join("")}</span>
              {genTokens.map((t, i) => (
                <Tooltip key={i} title={`P = ${(t.prob * 100).toFixed(1)}%`}>
                  <span
                    className="gen-token"
                    style={{ opacity: 0.5 + t.prob * 0.5, borderBottomColor: `hsl(${t.prob * 120}, 70%, 50%)` }}
                  >
                    {t.token}
                  </span>
                </Tooltip>
              ))}
              {isGenerating && <span className="gen-cursor">|</span>}
            </div>
          </Card>

          {/* Token probabilities */}
          {currentProbs && (
            <Card bordered={false} className="exp-probs">
              <Title level={5}>
                <MessageOutlined style={{ marginRight: 8 }} />
                Next token probabilities {selectedToken && <Tag color="green">Selected: "{selectedToken.token}"</Tag>}
              </Title>
              <Paragraph type="secondary" style={{ fontSize: 13 }}>
                {isGenerating
                  ? "Auto-selecting based on temperature sampling..."
                  : "Click a token to select it (or use Auto Generate for temperature-based sampling)"}
              </Paragraph>
              <div className="prob-bars">
                {currentProbs.map((p, i) => (
                  <div
                    key={i}
                    className={`prob-bar-item ${selectedToken?.token === p.token ? "selected" : ""} ${!isGenerating ? "clickable" : ""}`}
                    onClick={() => !isGenerating && pickToken(p)}
                  >
                    <div className="prob-token-label">
                      <code>"{p.token}"</code>
                    </div>
                    <div className="prob-bar-track">
                      <div
                        className="prob-bar-fill"
                        style={{ width: `${p.prob * 100}%`, background: `hsl(${p.prob * 120}, 70%, 45%)` }}
                      />
                    </div>
                    <div className="prob-value">{(p.prob * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Explanation */}
          <Card bordered={false} className="exp-explain">
            <Title level={5}><BulbOutlined /> What's happening</Title>
            <div className="exp-explain-steps">
              <div className="exp-step">
                <Tag color="blue">1</Tag>
                <Text>Your prompt goes through all 96 transformer layers (attention + FFN)</Text>
              </div>
              <div className="exp-step">
                <Tag color="purple">2</Tag>
                <Text>The output layer produces a probability for EVERY token in the vocabulary (~100K tokens)</Text>
              </div>
              <div className="exp-step">
                <Tag color="green">3</Tag>
                <Text>Temperature scales these probabilities (low = peaked, high = flat)</Text>
              </div>
              <div className="exp-step">
                <Tag color="orange">4</Tag>
                <Text>One token is sampled, appended to context, and the whole process repeats</Text>
              </div>
            </div>
          </Card>

          <div className="lesson-nav-btns">
            <Button icon={<BookOutlined />} onClick={() => setMode("learn")}>Back to Lesson</Button>
            <Button type="primary" icon={<CheckOutlined />} onClick={() => setMode("quiz")}>
              Take the Quiz
            </Button>
          </div>
        </div>
      )}

      {/* Quiz mode */}
      {mode === "quiz" && (
        <div className="llm-quiz">
          <Alert
            type="info"
            showIcon
            icon={<BulbOutlined />}
            message="Answer 5 questions about LLMs. Score 3/5 or higher to complete this module."
            style={{ marginBottom: 20 }}
          />

          {QUIZ.map((q, qIdx) => (
            <Card key={qIdx} bordered={false} className="quiz-question-card">
              <div className="quiz-q-header">
                <Tag color="blue">Q{qIdx + 1}</Tag>
                <Text strong>{q.question}</Text>
              </div>
              <Radio.Group
                value={quizAnswers[qIdx]}
                onChange={(e) => setQuizAnswers(prev => ({ ...prev, [qIdx]: e.target.value }))}
                disabled={quizSubmitted}
                className="quiz-options"
              >
                <Space direction="vertical" style={{ width: "100%" }}>
                  {q.options.map((opt, oIdx) => (
                    <Radio key={oIdx} value={oIdx} className="quiz-option">
                      {opt}
                    </Radio>
                  ))}
                </Space>
              </Radio.Group>

              {quizSubmitted && (
                <div className="quiz-feedback">
                  {quizAnswers[qIdx] === q.correct ? (
                    <Tag icon={<CheckCircleOutlined />} color="success">Correct!</Tag>
                  ) : (
                    <Tag icon={<CloseCircleOutlined />} color="error">
                      Incorrect — answer: {q.options[q.correct]}
                    </Tag>
                  )}
                  <Button
                    type="link"
                    size="small"
                    onClick={() => setShowSolution(prev => ({ ...prev, [qIdx]: !prev[qIdx] }))}
                  >
                    {showSolution[qIdx] ? "Hide" : "Show"} explanation
                  </Button>
                  {showSolution[qIdx] && (
                    <div className="quiz-explanation">
                      <BulbOutlined style={{ color: "#fbbf24", marginRight: 6 }} />
                      {q.explanation}
                    </div>
                  )}
                </div>
              )}
            </Card>
          ))}

          {!quizSubmitted && (
            <div className="quiz-submit">
              <Button
                type="primary"
                size="large"
                onClick={submitQuiz}
                disabled={Object.keys(quizAnswers).length < 5}
                icon={<CheckOutlined />}
              >
                Submit Answers ({Object.keys(quizAnswers).length}/5 answered)
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Result mode */}
      {mode === "result" && (
        <div className="llm-result">
          <Result
            status={passed ? "success" : "warning"}
            icon={passed ? <TrophyOutlined /> : undefined}
            title={passed ? "LLM Module Complete!" : "Not quite — try again"}
            subTitle={`You scored ${quizScore}/5 ${passed ? "— you understand how LLMs generate text!" : "— need at least 3/5"}`}
            extra={
              <Space>
                {passed && (
                  <Button type="primary" icon={<ArrowRightOutlined />} onClick={() => setMode("learn")}>
                    Review or Continue
                  </Button>
                )}
                {!passed && (
                  <Button icon={<ReloadOutlined />} onClick={retryQuiz}>
                    Retry Quiz
                  </Button>
                )}
                <Button onClick={() => setMode("experiment")}>
                  <ExperimentOutlined /> Back to Experiment
                </Button>
              </Space>
            }
          />
        </div>
      )}
    </div>
  );
}
