import { useState, useEffect, useMemo } from "react";
import { Card, Button, Steps, Typography, Tag, Space, Radio, Alert, Progress, Result, Divider, Row, Col, Statistic } from "antd";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  LockOutlined,
  TrophyOutlined,
  ArrowRightOutlined,
  ArrowLeftOutlined,
  ReloadOutlined,
  BulbOutlined,
  ExperimentOutlined,
  BookOutlined,
  CheckOutlined,
} from "@ant-design/icons";
import {
  traceMatMul,
  traceSoftmax,
  traceReLU,
  traceGELU,
  traceTokenize,
  traceAttention,
} from "../engine/nn_wasm";
import StepViewer from "./StepViewer";
import "./MathExplorer.css";

const { Title, Paragraph, Text } = Typography;

// ═══════════════════════════════════════════════════════════════════
// TOPIC DATA — 6 sequential topics, each with lesson + 5 quiz questions
// ═══════════════════════════════════════════════════════════════════

const TOPICS = [
  {
    id: "tokenize",
    title: "Tokenization",
    icon: "✂",
    color: "#fbbf24",
    subtitle: "Text → Numbers",
    lesson: {
      definition: "Tokenization is the very first step. AI models can't read text — they only understand numbers. A tokenizer splits text into pieces (tokens) and assigns each piece a number from a fixed vocabulary.",
      formula: '"Hello world" → ["Hello", " world"] → [15496, 995]',
      keyPoints: [
        "Every word or subword gets a unique integer ID",
        "Common words are 1 token, rare words split into subwords",
        "GPT-4 vocabulary has ~100,000 tokens",
        "This is always the FIRST step before any computation",
      ],
      example: {
        input: '"The cat sat"',
        steps: ['Split: ["The", " cat", " sat"]', "Lookup IDs: [464, 3797, 3332]", "These IDs index into an embedding table"],
        output: "[464, 3797, 3332]",
      },
    },
    quiz: [
      {
        question: 'What does tokenization convert?',
        options: ["Numbers to text", "Text to numbers", "Images to text", "Numbers to images"],
        correct: 1,
        explanation: "Tokenization converts raw text into integer IDs that neural networks can process.",
      },
      {
        question: 'Why can\'t AI models read text directly?',
        options: ["They're too slow", "They only work with numbers (tensors)", "Text is too long", "They can — tokenization is optional"],
        correct: 1,
        explanation: "Neural networks perform mathematical operations (matrix multiply, etc.) which require numerical inputs, not characters.",
      },
      {
        question: 'The word "unhappiness" might be tokenized as:',
        options: ['["unhappiness"] (1 token)', '["un", "happiness"] (2 tokens)', '["un", "happi", "ness"] (3 tokens)', "Any of the above depending on the vocabulary"],
        correct: 3,
        explanation: "How a word is split depends on the specific tokenizer's vocabulary. BPE-based tokenizers learn splits from data.",
      },
      {
        question: "If GPT-4 has ~100K tokens in its vocabulary, what does that mean?",
        options: ["It can only process 100K words", "Each token ID is a number from 0 to ~99,999", "It has 100K parameters", "It can only generate 100K characters"],
        correct: 1,
        explanation: "The vocabulary size means each token is mapped to an integer in range [0, vocab_size-1]. The model then uses these IDs to look up embeddings.",
      },
      {
        question: "What comes AFTER tokenization in an AI pipeline?",
        options: ["Output generation", "Embedding lookup (token IDs → vectors)", "Softmax", "Image processing"],
        correct: 1,
        explanation: "After getting integer IDs, the model looks up a learned vector (embedding) for each ID. This gives each token a rich numerical representation.",
      },
    ],
  },
  {
    id: "matmul",
    title: "Matrix Multiply",
    icon: "×",
    color: "#4a90e2",
    subtitle: "The Core Operation",
    lesson: {
      definition: "Matrix multiplication is THE fundamental operation of neural networks. Every prediction a model makes comes from multiplying matrices together. It combines inputs with learned weights to produce outputs.",
      formula: "C[i][j] = Σₖ A[i][k] × B[k][j]",
      keyPoints: [
        "Each output cell = dot product of a row from A and a column from B",
        "A[M×K] × B[K×N] = C[M×N] — inner dimensions must match",
        "Every linear layer is a matrix multiply",
        "~90% of neural network compute time is spent here",
      ],
      example: {
        input: "A = [[1,2],[3,4]], B = [[5,6],[7,8]]",
        steps: [
          "C[0][0] = 1×5 + 2×7 = 5 + 14 = 19",
          "C[0][1] = 1×6 + 2×8 = 6 + 16 = 22",
          "C[1][0] = 3×5 + 4×7 = 15 + 28 = 43",
          "C[1][1] = 3×6 + 4×8 = 18 + 32 = 50",
        ],
        output: "C = [[19, 22], [43, 50]]",
      },
    },
    quiz: [
      {
        question: "What is [1, 2, 3] · [4, 5, 6] (dot product)?",
        options: ["15", "32", "21", "45"],
        correct: 1,
        explanation: "Dot product = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32",
      },
      {
        question: "If A is [2×3] and B is [3×4], what shape is A×B?",
        options: ["[2×4]", "[3×3]", "[2×3]", "[4×2]"],
        correct: 0,
        explanation: "A[M×K] × B[K×N] = C[M×N]. Here M=2, K=3, N=4, so result is [2×4].",
      },
      {
        question: "Can you multiply A[2×3] × B[2×3]?",
        options: ["Yes, result is [2×3]", "Yes, result is [3×2]", "No — inner dimensions don't match (3 ≠ 2)", "Yes, result is [2×2]"],
        correct: 2,
        explanation: "For A[M×K] × B[K'×N], we need K = K'. Here K=3 but K'=2, so the multiply is impossible.",
      },
      {
        question: "In A = [[1,0],[0,1]] × B = [[5,6],[7,8]], what is the result?",
        options: ["[[5,6],[7,8]]", "[[12,14],[5,6]]", "[[5,7],[6,8]]", "[[0,0],[0,0]]"],
        correct: 0,
        explanation: "A is the identity matrix. Multiplying any matrix by the identity gives back the same matrix. I × B = B.",
      },
      {
        question: "Why is matrix multiply so important for AI?",
        options: ["It's fast to compute", "Every linear/dense layer is a matrix multiply of inputs × weights", "It's the only math operation AI uses", "It compresses data"],
        correct: 1,
        explanation: "A neural network layer computes output = input × weights + bias. The input × weights step is matrix multiplication.",
      },
    ],
  },
  {
    id: "relu",
    title: "ReLU",
    icon: "⌐",
    color: "#34d399",
    subtitle: "max(0, x)",
    lesson: {
      definition: "ReLU (Rectified Linear Unit) is the simplest activation function: it keeps positive values as-is and replaces negatives with zero. Without activation functions, stacking layers would be useless — multiple linear layers are mathematically equivalent to one linear layer.",
      formula: "ReLU(x) = max(0, x)",
      keyPoints: [
        "Positive → stays the same. Negative → becomes 0",
        "Introduces non-linearity (lets networks learn curves, not just lines)",
        "Without ReLU: 100 layers = 1 layer (just one big matrix multiply)",
        "Creates sparsity — many zeros, which helps networks learn distinct features",
      ],
      example: {
        input: "[-2, -0.5, 0, 0.3, 1.5, -1.2]",
        steps: [
          "max(0, -2) = 0",
          "max(0, -0.5) = 0",
          "max(0, 0) = 0",
          "max(0, 0.3) = 0.3",
          "max(0, 1.5) = 1.5",
          "max(0, -1.2) = 0",
        ],
        output: "[0, 0, 0, 0.3, 1.5, 0]",
      },
    },
    quiz: [
      {
        question: "What is ReLU(-5)?",
        options: ["-5", "5", "0", "1"],
        correct: 2,
        explanation: "ReLU(x) = max(0, x). Since -5 < 0, ReLU(-5) = 0.",
      },
      {
        question: "What is ReLU(3.7)?",
        options: ["0", "1", "3.7", "-3.7"],
        correct: 2,
        explanation: "ReLU(x) = max(0, x). Since 3.7 > 0, it passes through unchanged. ReLU(3.7) = 3.7.",
      },
      {
        question: "Why do neural networks NEED activation functions like ReLU?",
        options: ["To make computation faster", "Without them, many layers collapse into one (no non-linearity)", "To reduce memory usage", "They're optional — just improve accuracy slightly"],
        correct: 1,
        explanation: "Without non-linearity, layer1(layer2(x)) = single_layer(x). Activations break this linearity, letting networks learn complex patterns.",
      },
      {
        question: "Apply ReLU to [-1, 2, -3, 4, 0]. How many zeros in the output?",
        options: ["1", "2", "3", "4"],
        correct: 2,
        explanation: "ReLU turns negatives to 0 and keeps 0 as 0. Input has -1, -3, 0 → output zeros at positions 0, 2, 4. That's 3 zeros.",
      },
      {
        question: "What is the main disadvantage of ReLU?",
        options: ["It's too complex to compute", '"Dead neurons" — once a neuron outputs 0, it may never recover', "It only works on images", "It changes the shape of the data"],
        correct: 1,
        explanation: "If a neuron consistently gets negative inputs, its gradient is always 0 and it stops learning. This is called the 'dying ReLU' problem.",
      },
    ],
  },
  {
    id: "gelu",
    title: "GELU",
    icon: "≈",
    color: "#fb923c",
    subtitle: "Smooth Activation",
    lesson: {
      definition: "GELU (Gaussian Error Linear Unit) is a smoother version of ReLU used in modern transformers (GPT, BERT). Instead of a hard cutoff at 0, it gradually reduces small negative values. Think of it as 'probabilistic ReLU' — it weights values by how likely they are to be positive.",
      formula: "GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))",
      keyPoints: [
        "Smooth — no hard edge at 0 (unlike ReLU)",
        "Small negatives get slightly reduced, not killed entirely",
        "Used in GPT-2, GPT-3, GPT-4, BERT, and most modern models",
        "Better training dynamics than ReLU for transformer architectures",
      ],
      example: {
        input: "[-2, -1, -0.5, 0, 0.5, 1, 2]",
        steps: [
          "GELU(-2) ≈ -0.045 (almost 0, but not exactly)",
          "GELU(-1) ≈ -0.159 (slightly negative, not killed)",
          "GELU(-0.5) ≈ -0.154",
          "GELU(0) = 0 (same as ReLU at 0)",
          "GELU(0.5) ≈ 0.346 (slightly less than 0.5)",
          "GELU(1) ≈ 0.841",
          "GELU(2) ≈ 1.955 (almost same as input for large positives)",
        ],
        output: "[-0.045, -0.159, -0.154, 0, 0.346, 0.841, 1.955]",
      },
    },
    quiz: [
      {
        question: "How does GELU differ from ReLU for input x = -0.5?",
        options: ["Both output 0", "ReLU outputs 0, GELU outputs a small negative value", "Both output -0.5", "GELU outputs 0, ReLU outputs -0.5"],
        correct: 1,
        explanation: "ReLU(-0.5) = 0 (hard cutoff). GELU(-0.5) ≈ -0.154 (smooth reduction). GELU doesn't fully kill small negatives.",
      },
      {
        question: "For large positive values (like x=10), GELU behaves like:",
        options: ["It outputs 0", "It outputs exactly 10 (identity)", "It outputs approximately 10 (nearly identity)", "It outputs 1"],
        correct: 2,
        explanation: "For large positive x, GELU(x) ≈ x. The function approaches the identity for large positives, similar to ReLU.",
      },
      {
        question: "Why did GPT-2/3/4 switch from ReLU to GELU?",
        options: ["GELU is faster to compute", "GELU's smooth gradients improve training stability for transformers", "ReLU doesn't work on text", "GELU uses less memory"],
        correct: 1,
        explanation: "GELU provides smoother gradients (no abrupt zero-slope region), which helps optimization converge better in large transformer models.",
      },
      {
        question: "GELU(0) equals:",
        options: ["0", "0.5", "1", "-1"],
        correct: 0,
        explanation: "At x=0, GELU(0) = 0.5 × 0 × (1 + tanh(0)) = 0. The function passes through the origin.",
      },
      {
        question: "Which statement about GELU is FALSE?",
        options: ["It's used in GPT and BERT", "It's smoother than ReLU", "It always outputs positive values", "It's more expensive to compute than ReLU"],
        correct: 2,
        explanation: "GELU can output small negative values (e.g., GELU(-1) ≈ -0.159). It does NOT force all outputs to be ≥ 0 like ReLU does.",
      },
    ],
  },
  {
    id: "softmax",
    title: "Softmax",
    icon: "σ",
    color: "#a78bfa",
    subtitle: "Scores → Probabilities",
    lesson: {
      definition: "Softmax converts a vector of arbitrary numbers (called logits or scores) into a probability distribution — all values become positive and sum to exactly 1.0. It's like asking 'given these scores, what's the probability of each class?' The highest score gets a disproportionately high probability.",
      formula: "softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ",
      keyPoints: [
        "Output values are all between 0 and 1",
        "All outputs sum to exactly 1.0 (valid probability distribution)",
        "Amplifies differences — 'winner takes more'",
        "Used in attention weights AND final prediction layer",
      ],
      example: {
        input: "[2.0, 1.0, 0.1]",
        steps: [
          "e^2.0 = 7.389",
          "e^1.0 = 2.718",
          "e^0.1 = 1.105",
          "Sum = 7.389 + 2.718 + 1.105 = 11.212",
          "softmax = [7.389/11.212, 2.718/11.212, 1.105/11.212]",
        ],
        output: "[0.659, 0.242, 0.099] — sums to 1.0",
      },
    },
    quiz: [
      {
        question: "What must ALL softmax outputs sum to?",
        options: ["0", "The input sum", "1.0", "It varies"],
        correct: 2,
        explanation: "Softmax always produces a valid probability distribution where all values sum to exactly 1.0.",
      },
      {
        question: "If input is [10, 0, 0], softmax gives approximately:",
        options: ["[0.33, 0.33, 0.33]", "[1.0, 0.0, 0.0]", "[10, 0, 0]", "[0.5, 0.25, 0.25]"],
        correct: 1,
        explanation: "When one value is much larger, softmax gives it nearly all the probability. e^10 >> e^0, so the first element dominates (~0.9999).",
      },
      {
        question: "If input is [1, 1, 1, 1], what does softmax output?",
        options: ["[1, 1, 1, 1]", "[0.25, 0.25, 0.25, 0.25]", "[0, 0, 0, 0]", "[4, 4, 4, 4]"],
        correct: 1,
        explanation: "When all inputs are equal, all e^x values are equal, so each gets 1/n of the total. With 4 equal inputs: [0.25, 0.25, 0.25, 0.25].",
      },
      {
        question: "Can softmax output a negative number?",
        options: ["Yes, for negative inputs", "No — all outputs are between 0 and 1", "Yes, but only for very large negative inputs", "Sometimes"],
        correct: 1,
        explanation: "Since e^x is always positive, and we divide by a positive sum, softmax outputs are always in (0, 1).",
      },
      {
        question: "Where is softmax used in a transformer?",
        options: ["Only in the final output layer", "Only in attention", "In both attention (Q·K scores) AND the final output (logits → word probabilities)", "Nowhere — transformers use sigmoid"],
        correct: 2,
        explanation: "Softmax appears twice: (1) converting attention scores to weights, and (2) converting final logits to next-token probabilities.",
      },
    ],
  },
  {
    id: "attention",
    title: "Attention",
    icon: "◎",
    color: "#22d3ee",
    subtitle: "The Key Innovation",
    lesson: {
      definition: 'Self-Attention is the breakthrough that made transformers work. Each token looks at every other token, computes a relevance score, and creates a new representation that "understands context." This is how "bank" knows if you mean river bank or money bank — by attending to surrounding words.',
      formula: "Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V",
      keyPoints: [
        "Q (Query) = 'what am I looking for?'",
        "K (Key) = 'what do I contain?'",
        "V (Value) = 'here's my information'",
        "Q·K dot product = how relevant two tokens are to each other",
        "√d scaling prevents scores from getting too large",
        "Softmax converts scores to probabilities (attention weights)",
        "Final multiply by V = weighted mix of all token values",
      ],
      example: {
        input: '"The cat sat on the mat" — processing the word "sat"',
        steps: [
          '"sat" creates a Query: "what did I do? what\'s my subject?"',
          'Each word creates a Key: "cat"→"I\'m a noun/subject", "on"→"I\'m a preposition"',
          'Q·K scores: "sat"·"cat" = high (subject-verb!), "sat"·"the" = low',
          "Softmax: [0.05, 0.60, 0.05, 0.15, 0.05, 0.10] (cat gets most attention)",
          'Multiply by Values: "sat" now encodes "I\'m an action done by the cat"',
        ],
        output: "A new vector for 'sat' that encodes its relationship to all other words",
      },
    },
    quiz: [
      {
        question: "In Attention(Q, K, V), what does Q × Kᵀ compute?",
        options: ["The final output", "Relevance scores between all token pairs", "The token embeddings", "The loss function"],
        correct: 1,
        explanation: "Q × Kᵀ computes dot products between every query-key pair, giving a score of how much each token should attend to every other token.",
      },
      {
        question: "Why divide by √d in the attention formula?",
        options: ["To save memory", "To prevent dot products from growing too large (which makes softmax too peaked)", "It's optional — just for speed", "To normalize to unit length"],
        correct: 1,
        explanation: "As embedding dimension d grows, dot products get larger. Dividing by √d keeps values in a reasonable range so softmax produces useful gradients.",
      },
      {
        question: "If sequence length is n, how many attention scores does Q×Kᵀ produce?",
        options: ["n", "n × n", "n × d", "d × d"],
        correct: 1,
        explanation: "Every token attends to every other token, so Q×Kᵀ produces an n×n matrix of scores. This is why attention is O(n²) — expensive for long sequences.",
      },
      {
        question: 'In "The bank by the river", what helps the model know "bank" means river bank?',
        options: ["Tokenization", "Matrix multiply alone", "Attention — 'bank' attends to 'river' and gets high score", "ReLU activation"],
        correct: 2,
        explanation: "Attention lets 'bank' look at all other words. The high attention score with 'river' tells the model this is a river bank, not a financial bank.",
      },
      {
        question: "What is Multi-Head Attention?",
        options: ["Running attention multiple times with different learned Q/K/V projections", "Using multiple transformer layers", "Attending to multiple sentences at once", "A faster version of attention"],
        correct: 0,
        explanation: "Multi-head attention runs several attention operations in parallel, each with different learned weight matrices. This lets the model attend to different types of relationships simultaneously (e.g., one head for syntax, another for semantics).",
      },
    ],
  },
];

// ═══════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════

export default function MathExplorer() {
  // Track which topics are completed (stored in localStorage)
  const [completedTopics, setCompletedTopics] = useState(() => {
    try {
      const saved = localStorage.getItem("inferx-math-completed");
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });

  const [currentTopicIdx, setCurrentTopicIdx] = useState(() => {
    // Start at the first uncompleted topic
    try {
      const saved = localStorage.getItem("inferx-math-completed");
      const completed = saved ? JSON.parse(saved) : [];
      const nextIdx = TOPICS.findIndex(t => !completed.includes(t.id));
      return nextIdx >= 0 ? nextIdx : 0;
    } catch { return 0; }
  });

  const [mode, _setMode] = useState("lesson"); // "lesson" | "quiz" | "result"
  const topicContentRef = useMemo(() => ({ current: null }), []);

  // Scroll to top of topic content when mode changes
  const setMode = (newMode) => {
    _setMode(newMode);
    // Scroll into view after React re-renders
    setTimeout(() => {
      if (topicContentRef.current) {
        topicContentRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
      } else {
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    }, 50);
  };
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizSubmitted, setQuizSubmitted] = useState(false);
  const [showSolution, setShowSolution] = useState({});

  // Interactive demo state
  const [traceSteps, setTraceSteps] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);

  const currentTopic = TOPICS[currentTopicIdx];

  // Persist completions
  useEffect(() => {
    try { localStorage.setItem("inferx-math-completed", JSON.stringify(completedTopics)); } catch {}
  }, [completedTopics]);

  // Check if a topic is unlocked (previous must be completed, or it's the first)
  function isUnlocked(idx) {
    if (idx === 0) return true;
    return completedTopics.includes(TOPICS[idx - 1].id);
  }

  // Select a topic
  function selectTopic(idx) {
    if (!isUnlocked(idx)) return;
    setCurrentTopicIdx(idx);
    setMode("lesson");
    setQuizAnswers({});
    setQuizSubmitted(false);
    setShowSolution({});
    setTraceSteps([]);
  }

  // Submit quiz
  function submitQuiz() {
    setQuizSubmitted(true);
    const correct = currentTopic.quiz.filter((q, i) => quizAnswers[i] === q.correct).length;
    // Pass if >= 3/5 correct
    if (correct >= 3 && !completedTopics.includes(currentTopic.id)) {
      setCompletedTopics(prev => [...prev, currentTopic.id]);
    }
    setMode("result");
  }

  // Reset quiz
  function retryQuiz() {
    setQuizAnswers({});
    setQuizSubmitted(false);
    setShowSolution({});
    setMode("quiz");
  }

  // Move to next topic
  function goNext() {
    if (currentTopicIdx < TOPICS.length - 1) {
      selectTopic(currentTopicIdx + 1);
    }
  }

  // Run interactive demo
  function runDemo() {
    let steps = [];
    switch (currentTopic.id) {
      case "tokenize":
        steps = traceTokenize("Hello world");
        break;
      case "matmul":
        steps = traceMatMul([1, 2, 3, 4], 2, 2, [5, 6, 7, 8], 2, 2);
        break;
      case "relu":
        steps = traceReLU([-2, -0.5, 0, 0.3, 1.5, -1.2]);
        break;
      case "gelu":
        steps = traceGELU([-2, -1, -0.5, 0, 0.5, 1, 2]);
        break;
      case "softmax":
        steps = traceSoftmax([2.0, 1.0, 0.1]);
        break;
      case "attention":
        steps = traceAttention(4);
        break;
    }
    setTraceSteps(steps || []);
    setCurrentStep(0);
  }

  // Quiz score calculation
  const quizScore = useMemo(() => {
    if (!quizSubmitted) return 0;
    return currentTopic.quiz.filter((q, i) => quizAnswers[i] === q.correct).length;
  }, [quizSubmitted, quizAnswers, currentTopic]);

  const passed = quizScore >= 3;

  // Overall progress
  const overallProgress = Math.round((completedTopics.length / TOPICS.length) * 100);

  return (
    <div className="math-explorer animate-in">
      {/* ═══ Header ═══ */}
      <div className="math-header">
        <Title level={2} style={{ marginBottom: 4 }}>
          <span style={{ marginRight: 8 }}>∑</span>Math Lab
        </Title>
        <Paragraph type="secondary">
          Master the 6 operations that power every AI model. Complete each topic to unlock the next.
        </Paragraph>
        <Progress
          percent={overallProgress}
          steps={6}
          strokeColor="#4a90e2"
          size="small"
          format={() => `${completedTopics.length}/6`}
        />
      </div>

      {/* ═══ Topic Navigation (Steps) ═══ */}
      <div className="math-steps-nav">
        <Steps
          current={currentTopicIdx}
          size="small"
          onChange={(idx) => selectTopic(idx)}
          items={TOPICS.map((topic, idx) => ({
            title: topic.title,
            icon: completedTopics.includes(topic.id)
              ? <CheckCircleOutlined style={{ color: "#34d399" }} />
              : !isUnlocked(idx)
                ? <LockOutlined style={{ color: "#64748b" }} />
                : undefined,
            disabled: !isUnlocked(idx),
            status: completedTopics.includes(topic.id)
              ? "finish"
              : idx === currentTopicIdx
                ? "process"
                : !isUnlocked(idx)
                  ? "wait"
                  : "wait",
          }))}
        />
      </div>

      {/* ═══ Topic Content ═══ */}
      <div className="math-topic-content" ref={(el) => { topicContentRef.current = el; }}>
        {/* Topic Header */}
        <div className="topic-header" style={{ borderColor: currentTopic.color }}>
          <div className="topic-icon" style={{ background: `${currentTopic.color}22`, color: currentTopic.color }}>
            {currentTopic.icon}
          </div>
          <div>
            <Title level={3} style={{ marginBottom: 0 }}>
              {currentTopicIdx + 1}. {currentTopic.title}
            </Title>
            <Text type="secondary">{currentTopic.subtitle}</Text>
          </div>
          <div className="topic-mode-switch">
            <Space>
              <Button
                type={mode === "lesson" ? "primary" : "default"}
                icon={<BookOutlined />}
                onClick={() => setMode("lesson")}
              >
                Learn
              </Button>
              <Button
                type={mode === "quiz" || mode === "result" ? "primary" : "default"}
                icon={<ExperimentOutlined />}
                onClick={() => setMode("quiz")}
              >
                Quiz
              </Button>
            </Space>
          </div>
        </div>

        {/* ═══ LESSON MODE ═══ */}
        {mode === "lesson" && (
          <div className="topic-lesson">
            {/* Definition */}
            <Card className="lesson-card" bordered={false}>
              <Title level={5}><BookOutlined /> What is it?</Title>
              <Paragraph>{currentTopic.lesson.definition}</Paragraph>
            </Card>

            {/* Formula */}
            <Card className="lesson-card formula-card" bordered={false}>
              <Title level={5}>Formula</Title>
              <div className="formula-display">
                <code>{currentTopic.lesson.formula}</code>
              </div>
            </Card>

            {/* Key Points */}
            <Card className="lesson-card" bordered={false}>
              <Title level={5}><BulbOutlined /> Key Points</Title>
              <ul className="key-points-list">
                {currentTopic.lesson.keyPoints.map((point, i) => (
                  <li key={i}>{point}</li>
                ))}
              </ul>
            </Card>

            {/* Worked Example */}
            <Card className="lesson-card example-card" bordered={false}>
              <Title level={5}><ArrowRightOutlined /> Worked Example</Title>
              <div className="example-section">
                <div className="example-row">
                  <Tag color="blue">Input</Tag>
                  <code>{currentTopic.lesson.example.input}</code>
                </div>
                <div className="example-steps">
                  {currentTopic.lesson.example.steps.map((step, i) => (
                    <div key={i} className="example-step">
                      <span className="step-num">{i + 1}</span>
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
                <div className="example-row">
                  <Tag color="green">Output</Tag>
                  <code>{currentTopic.lesson.example.output}</code>
                </div>
              </div>
            </Card>

            {/* Interactive Demo */}
            <Card className="lesson-card" bordered={false}>
              <Title level={5}><ExperimentOutlined /> Try It Live</Title>
              <Paragraph type="secondary">Run the operation and see step-by-step computation trace from our C++ engine.</Paragraph>
              <Button type="primary" icon={<ExperimentOutlined />} onClick={runDemo}>
                Run {currentTopic.title}
              </Button>
              {traceSteps.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <StepViewer
                    steps={traceSteps}
                    currentStep={currentStep}
                    onStepChange={setCurrentStep}
                    title={currentTopic.title}
                  />
                </div>
              )}
            </Card>

            {/* CTA to quiz */}
            <div className="lesson-cta">
              <Button type="primary" size="large" icon={<ArrowRightOutlined />} onClick={() => setMode("quiz")}>
                Ready? Take the Quiz ({currentTopic.title})
              </Button>
            </div>
          </div>
        )}

        {/* ═══ QUIZ MODE ═══ */}
        {mode === "quiz" && (
          <div className="topic-quiz">
            <Alert
              type="info"
              showIcon
              icon={<BulbOutlined />}
              message={`Answer 5 questions about ${currentTopic.title}. Score 3/5 or higher to unlock the next topic.`}
              style={{ marginBottom: 20 }}
            />

            {currentTopic.quiz.map((q, qIdx) => (
              <Card key={qIdx} className="quiz-question-card" bordered={false}>
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

                {/* Show solution toggle */}
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

            {/* Submit button */}
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

        {/* ═══ RESULT MODE ═══ */}
        {mode === "result" && (
          <div className="topic-result">
            <Result
              status={passed ? "success" : "warning"}
              icon={passed ? <TrophyOutlined /> : undefined}
              title={passed ? `${currentTopic.title} Mastered!` : "Not quite — try again"}
              subTitle={`You scored ${quizScore}/5 ${passed ? "(3+ needed to pass)" : "— need at least 3/5 to unlock the next topic"}`}
              extra={
                <Space>
                  {passed && currentTopicIdx < TOPICS.length - 1 && (
                    <Button type="primary" icon={<ArrowRightOutlined />} onClick={goNext}>
                      Next: {TOPICS[currentTopicIdx + 1].title}
                    </Button>
                  )}
                  {passed && currentTopicIdx === TOPICS.length - 1 && (
                    <Tag icon={<TrophyOutlined />} color="gold" style={{ fontSize: 14, padding: "4px 12px" }}>
                      All topics completed! You've mastered the math of AI.
                    </Tag>
                  )}
                  {!passed && (
                    <Button icon={<ReloadOutlined />} onClick={retryQuiz}>
                      Retry Quiz
                    </Button>
                  )}
                  <Button onClick={() => setMode("lesson")}>
                    <BookOutlined /> Review Lesson
                  </Button>
                </Space>
              }
            />

            {/* Show answers review */}
            <Divider>Your Answers</Divider>
            <div className="result-review">
              {currentTopic.quiz.map((q, i) => (
                <div key={i} className={`result-item ${quizAnswers[i] === q.correct ? "correct" : "incorrect"}`}>
                  <Space>
                    {quizAnswers[i] === q.correct
                      ? <CheckCircleOutlined style={{ color: "#34d399" }} />
                      : <CloseCircleOutlined style={{ color: "#f87171" }} />
                    }
                    <Text>Q{i + 1}: {q.question}</Text>
                  </Space>
                  {quizAnswers[i] !== q.correct && (
                    <div className="result-correct-answer">
                      <Text type="secondary">Correct: {q.options[q.correct]}</Text>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ═══ Bottom Navigation ═══ */}
      <div className="math-bottom-nav">
        <Button
          disabled={currentTopicIdx === 0}
          icon={<ArrowLeftOutlined />}
          onClick={() => selectTopic(currentTopicIdx - 1)}
        >
          Previous
        </Button>
        <Text type="secondary">
          Topic {currentTopicIdx + 1} of {TOPICS.length}
        </Text>
        <Button
          type="primary"
          disabled={currentTopicIdx === TOPICS.length - 1 || !isUnlocked(currentTopicIdx + 1)}
          onClick={() => selectTopic(currentTopicIdx + 1)}
        >
          Next <ArrowRightOutlined />
        </Button>
      </div>
    </div>
  );
}
