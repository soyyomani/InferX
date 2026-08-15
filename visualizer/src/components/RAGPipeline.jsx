import { useState, useEffect, useMemo } from "react";
import { Card, Button, Typography, Tag, Space, Radio, Alert, Steps, Result, Divider, Row, Col, Input, Tooltip, Progress } from "antd";
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
  SearchOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  RobotOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import "./RAGPipeline.css";

const { Title, Paragraph, Text } = Typography;
const { TextArea } = Input;

// ═══════════════════════════════════════════════════════════════════
// LESSON DATA
// ═══════════════════════════════════════════════════════════════════

const LESSON_SECTIONS = [
  {
    title: "Why RAG exists",
    content: `You just learned that LLMs generate text by predicting the next token. But here's the problem: 
    LLMs only know what they were trained on. Ask Claude about your company's internal docs? It has no idea. 
    Ask ChatGPT about a paper published yesterday? It can't know — it was trained months ago.`,
    highlight: "RAG = Retrieval-Augmented Generation. It gives LLMs access to knowledge they weren't trained on.",
    realWorld: [
      { scenario: "You ask Claude about your company's refund policy", without: "Claude halluccinates a generic answer or says 'I don't know'", with: "RAG retrieves your actual policy doc, Claude answers accurately" },
      { scenario: "You ask about a bug in your codebase", without: "ChatGPT guesses based on common patterns", with: "RAG finds the relevant source files, GPT explains the actual bug" },
      { scenario: "You ask about today's news", without: "LLM's training data is months old — wrong answer", with: "RAG fetches recent articles, LLM summarizes them" },
    ],
  },
  {
    title: "The 4-step pipeline",
    content: `RAG has exactly 4 steps. Every RAG system — from ChatGPT with browsing to enterprise search — follows this pattern.`,
    highlight: "Embed → Search → Augment → Generate. That's it.",
    pipeline: [
      { step: 1, name: "Embed the query", desc: "Convert the user's question into a vector (the same embeddings you learned about!)", icon: "🧮", detail: "Uses the same embedding model that converts tokens to vectors. Your question becomes a point in 4096-dimensional space." },
      { step: 2, name: "Search the vector database", desc: "Find documents whose embeddings are closest to the query embedding", icon: "🔍", detail: "Cosine similarity between query vector and all document vectors. Top-K nearest neighbors are retrieved." },
      { step: 3, name: "Augment the prompt", desc: "Stuff the retrieved documents into the LLM's context", icon: "📄", detail: "Template: 'Given this context: {retrieved_docs}\\n\\nAnswer the question: {user_query}'" },
      { step: 4, name: "Generate the answer", desc: "LLM generates a response grounded in the retrieved context", icon: "🤖", detail: "The LLM now has the relevant info in its context window. It generates an answer using BOTH its knowledge AND the documents." },
    ],
  },
  {
    title: "Embeddings & Vector Search",
    content: `Remember embeddings from the Text Pipeline? In RAG, they're used differently. 
    Instead of embedding tokens for attention, we embed entire DOCUMENTS and QUERIES into the same vector space. 
    Similar meanings = nearby vectors.`,
    highlight: "If two texts mean the same thing, their vectors will be close together — even if they use different words.",
    examples: [
      { text: "How do I return a product?", similar: "What is the refund process?", score: 0.94 },
      { text: "How do I return a product?", similar: "Product return policy", score: 0.91 },
      { text: "How do I return a product?", similar: "What's the weather today?", score: 0.12 },
      { text: "How do I return a product?", similar: "Shipping and delivery info", score: 0.45 },
    ],
  },
  {
    title: "When to use RAG vs Fine-tuning",
    content: `RAG isn't the only way to give an LLM new knowledge. You could also fine-tune the model 
    (retrain it on your data). But they solve different problems.`,
    highlight: "RAG = dynamic knowledge (changes often). Fine-tuning = behavior/style changes.",
    comparison: [
      { aspect: "Best for", rag: "Facts, docs, data that changes", finetune: "Tone, format, domain style" },
      { aspect: "Update speed", rag: "Instant (just update the database)", finetune: "Hours/days of retraining" },
      { aspect: "Cost", rag: "Cheap (just vector DB + retrieval)", finetune: "Expensive (GPU training)" },
      { aspect: "Accuracy", rag: "High if docs are good", finetune: "Can hallucinate trained facts" },
      { aspect: "Example", rag: "Customer support bot with product docs", finetune: "Making GPT write like your brand" },
    ],
  },
];

// ═══════════════════════════════════════════════════════════════════
// QUIZ DATA
// ═══════════════════════════════════════════════════════════════════

const QUIZ = [
  {
    question: "What problem does RAG solve?",
    options: [
      "Making LLMs generate faster",
      "Giving LLMs access to knowledge they weren't trained on",
      "Reducing the size of LLMs",
      "Making LLMs more creative",
    ],
    correct: 1,
    explanation: "LLMs only know their training data. RAG retrieves relevant documents at query time and puts them in context, so the LLM can answer about things it was never trained on.",
  },
  {
    question: "What are the 4 steps of a RAG pipeline (in order)?",
    options: [
      "Tokenize → Attend → Predict → Output",
      "Embed query → Search vectors → Augment prompt → Generate answer",
      "Search → Download → Summarize → Display",
      "Fine-tune → Embed → Search → Respond",
    ],
    correct: 1,
    explanation: "RAG: (1) Embed the user query into a vector, (2) Search the vector DB for similar documents, (3) Augment the prompt with retrieved docs, (4) Generate an answer grounded in that context.",
  },
  {
    question: "Why does 'How do I return a product?' match with 'What is the refund process?' in vector search?",
    options: [
      "They share the same keywords",
      "Their embedding vectors are close because they have similar MEANING (semantic similarity)",
      "They have the same number of words",
      "The database stores them in the same category",
    ],
    correct: 1,
    explanation: "Embedding models encode meaning, not just keywords. 'Return a product' and 'refund process' are semantically similar — they map to nearby points in vector space even with zero word overlap.",
  },
  {
    question: "When should you use RAG instead of fine-tuning?",
    options: [
      "When you want to change the model's writing style",
      "When your knowledge base changes frequently and you need up-to-date answers",
      "When you want the model to be more creative",
      "When you have very little data",
    ],
    correct: 1,
    explanation: "RAG is ideal for dynamic knowledge — you just update the vector database. Fine-tuning is better for changing behavior/style. If your docs change weekly, RAG wins; retraining weekly would be impractical.",
  },
  {
    question: "In the 'Augment' step, what actually happens?",
    options: [
      "The LLM is retrained on the documents",
      "Retrieved documents are inserted into the prompt context before the LLM generates",
      "The documents are displayed to the user directly",
      "The embeddings are modified",
    ],
    correct: 1,
    explanation: "Augmentation = template like 'Given this context: {docs}\\nAnswer: {question}'. The LLM sees the docs in its context window (just like you reading a document before answering a question).",
  },
];

// ═══════════════════════════════════════════════════════════════════
// INTERACTIVE EXPERIMENT: Mini RAG system
// ═══════════════════════════════════════════════════════════════════

// Mini knowledge base (simulated vector DB)
const KNOWLEDGE_BASE = [
  { id: 1, title: "Refund Policy", content: "Customers can request a full refund within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days.", embedding: [0.8, 0.9, 0.1, 0.2] },
  { id: 2, title: "Shipping Info", content: "Standard shipping takes 5-7 business days. Express shipping (2-day) is available for $12.99. Free shipping on orders over $50.", embedding: [0.2, 0.3, 0.9, 0.1] },
  { id: 3, title: "Product Warranty", content: "All electronics come with a 2-year manufacturer warranty. Extended warranty (5 years) available for $29.99. Warranty covers manufacturing defects only.", embedding: [0.7, 0.6, 0.2, 0.3] },
  { id: 4, title: "Account Settings", content: "To change your password, go to Settings > Security > Change Password. Two-factor authentication is available via SMS or authenticator app.", embedding: [0.1, 0.2, 0.3, 0.9] },
  { id: 5, title: "Size Guide", content: "Measure your chest, waist, and hips. S: 34-36 chest. M: 38-40 chest. L: 42-44 chest. XL: 46-48 chest. When in doubt, size up.", embedding: [0.4, 0.8, 0.5, 0.2] },
  { id: 6, title: "Return Process", content: "To initiate a return: 1) Log into your account, 2) Go to Order History, 3) Click 'Return Item', 4) Print the prepaid shipping label, 5) Drop off at any carrier location.", embedding: [0.85, 0.85, 0.15, 0.25] },
];

// Simulated query embeddings
const QUERY_EMBEDDINGS = {
  "how do i return something": [0.82, 0.88, 0.12, 0.22],
  "return": [0.8, 0.85, 0.15, 0.2],
  "refund": [0.83, 0.9, 0.1, 0.18],
  "shipping": [0.2, 0.3, 0.92, 0.1],
  "how long does delivery take": [0.22, 0.35, 0.88, 0.12],
  "warranty": [0.7, 0.6, 0.22, 0.3],
  "password": [0.1, 0.2, 0.3, 0.92],
  "size": [0.4, 0.75, 0.5, 0.2],
  "what size should i get": [0.42, 0.78, 0.48, 0.22],
  default: [0.5, 0.5, 0.5, 0.5],
};

function getQueryEmbedding(query) {
  const lower = query.toLowerCase().trim();
  // Check for keyword matches
  for (const [key, emb] of Object.entries(QUERY_EMBEDDINGS)) {
    if (lower.includes(key)) return emb;
  }
  // Generate a pseudo-embedding based on query hash
  const hash = lower.split("").reduce((a, c) => a + c.charCodeAt(0), 0);
  return [(hash % 100) / 100, ((hash * 7) % 100) / 100, ((hash * 13) % 100) / 100, ((hash * 19) % 100) / 100];
}

function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const magB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (magA * magB);
}

function searchDocuments(queryEmbedding, topK = 3) {
  const scored = KNOWLEDGE_BASE.map(doc => ({
    ...doc,
    similarity: cosineSimilarity(queryEmbedding, doc.embedding),
  }));
  scored.sort((a, b) => b.similarity - a.similarity);
  return scored.slice(0, topK);
}

function generateAnswer(query, docs) {
  // Simulated LLM response based on top document
  const topDoc = docs[0];
  if (topDoc.similarity > 0.8) {
    return `Based on our documentation: ${topDoc.content}`;
  } else if (topDoc.similarity > 0.6) {
    return `I found some relevant information. ${topDoc.content} Let me know if you need more specific details.`;
  } else {
    return `I couldn't find a highly relevant document for your question. The closest match was "${topDoc.title}" (${(topDoc.similarity * 100).toFixed(0)}% relevant). You might want to rephrase your question or check if this topic is covered in our knowledge base.`;
  }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════

export default function RAGPipeline({ onComplete }) {
  const [mode, setMode] = useState("learn"); // "learn" | "experiment" | "quiz" | "result"
  const [lessonStep, setLessonStep] = useState(0);
  const [quizAnswers, setQuizAnswers] = useState({});
  const [quizSubmitted, setQuizSubmitted] = useState(false);
  const [showSolution, setShowSolution] = useState({});

  // Experiment state
  const [query, setQuery] = useState("");
  const [ragStage, setRagStage] = useState(0); // 0=idle, 1=embedding, 2=searching, 3=augmenting, 4=generating, 5=done
  const [queryEmbedding, setQueryEmbedding] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [augmentedPrompt, setAugmentedPrompt] = useState("");
  const [generatedAnswer, setGeneratedAnswer] = useState("");

  // Scroll to top on mode change
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [mode]);

  // ─── Experiment Functions ─────────────────────────────────────
  function runRAG() {
    if (!query.trim()) return;

    // Stage 1: Embed
    setRagStage(1);
    setSearchResults([]);
    setAugmentedPrompt("");
    setGeneratedAnswer("");

    const emb = getQueryEmbedding(query);
    setQueryEmbedding(emb);

    setTimeout(() => {
      // Stage 2: Search
      setRagStage(2);
      const results = searchDocuments(emb);
      setSearchResults(results);

      setTimeout(() => {
        // Stage 3: Augment
        setRagStage(3);
        const context = results.map(r => `[${r.title}]: ${r.content}`).join("\n\n");
        const prompt = `Given this context:\n${context}\n\nUser question: ${query}\n\nAnswer:`;
        setAugmentedPrompt(prompt);

        setTimeout(() => {
          // Stage 4: Generate
          setRagStage(4);
          const answer = generateAnswer(query, results);

          setTimeout(() => {
            setGeneratedAnswer(answer);
            setRagStage(5);
          }, 800);
        }, 600);
      }, 600);
    }, 500);
  }

  function resetExperiment() {
    setQuery("");
    setRagStage(0);
    setQueryEmbedding(null);
    setSearchResults([]);
    setAugmentedPrompt("");
    setGeneratedAnswer("");
  }

  // ─── Quiz Functions ───────────────────────────────────────────
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

  // ═══════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════

  return (
    <div className="rag-pipeline animate-in">
      {/* Header */}
      <div className="rag-header">
        <Title level={2} style={{ marginBottom: 4 }}>
          <DatabaseOutlined style={{ marginRight: 8 }} />
          RAG Pipeline
        </Title>
        <Paragraph type="secondary">
          How Claude and ChatGPT answer questions about things they weren't trained on — by retrieving real documents first.
        </Paragraph>
        <Space className="rag-mode-tabs">
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

      {/* ═══ LEARN MODE ═══ */}
      {mode === "learn" && (
        <div className="rag-learn">
          <div className="lesson-nav">
            <Steps
              current={lessonStep}
              size="small"
              onChange={setLessonStep}
              items={LESSON_SECTIONS.map(s => ({ title: s.title }))}
            />
          </div>

          <div className="lesson-content">
            <Card bordered={false} className="lesson-main-card">
              <Title level={4}>{LESSON_SECTIONS[lessonStep].title}</Title>
              <Paragraph style={{ fontSize: 15, lineHeight: 1.8 }}>
                {LESSON_SECTIONS[lessonStep].content}
              </Paragraph>

              {LESSON_SECTIONS[lessonStep].highlight && (
                <Alert
                  type="info"
                  showIcon
                  icon={<BulbOutlined />}
                  message={LESSON_SECTIONS[lessonStep].highlight}
                  className="lesson-highlight"
                />
              )}

              {/* Real-world examples */}
              {LESSON_SECTIONS[lessonStep].realWorld && (
                <div className="real-world-examples">
                  <Title level={5}>Real Examples: With vs Without RAG</Title>
                  {LESSON_SECTIONS[lessonStep].realWorld.map((ex, i) => (
                    <div key={i} className="rw-example">
                      <div className="rw-scenario">
                        <Tag color="blue">Scenario</Tag>
                        <Text strong>{ex.scenario}</Text>
                      </div>
                      <Row gutter={12}>
                        <Col xs={24} md={12}>
                          <div className="rw-result rw-without">
                            <Tag color="red">Without RAG</Tag>
                            <Text type="secondary">{ex.without}</Text>
                          </div>
                        </Col>
                        <Col xs={24} md={12}>
                          <div className="rw-result rw-with">
                            <Tag color="green">With RAG</Tag>
                            <Text>{ex.with}</Text>
                          </div>
                        </Col>
                      </Row>
                    </div>
                  ))}
                </div>
              )}

              {/* Pipeline steps */}
              {LESSON_SECTIONS[lessonStep].pipeline && (
                <div className="pipeline-visual">
                  {LESSON_SECTIONS[lessonStep].pipeline.map((p, i) => (
                    <div key={i} className="pipeline-step-card">
                      <div className="pipeline-step-header">
                        <span className="pipeline-step-icon">{p.icon}</span>
                        <div>
                          <Text strong>Step {p.step}: {p.name}</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: 13 }}>{p.desc}</Text>
                        </div>
                      </div>
                      <div className="pipeline-step-detail">
                        <BulbOutlined style={{ color: "#fbbf24", marginRight: 6 }} />
                        <Text style={{ fontSize: 13 }}>{p.detail}</Text>
                      </div>
                      {i < 3 && <div className="pipeline-arrow">↓</div>}
                    </div>
                  ))}
                </div>
              )}

              {/* Similarity examples */}
              {LESSON_SECTIONS[lessonStep].examples && (
                <div className="similarity-section">
                  <Title level={5}>Semantic Similarity in Action</Title>
                  <Paragraph type="secondary">Same query compared against different texts. Notice: meaning matters more than keywords.</Paragraph>
                  {LESSON_SECTIONS[lessonStep].examples.map((ex, i) => (
                    <div key={i} className="sim-row">
                      <div className="sim-texts">
                        <code className="sim-query">"{ex.text}"</code>
                        <span className="sim-vs">vs</span>
                        <code className="sim-doc">"{ex.similar}"</code>
                      </div>
                      <div className="sim-score-bar">
                        <div className="sim-fill" style={{ width: `${ex.score * 100}%`, background: `hsl(${ex.score * 120}, 70%, 45%)` }} />
                      </div>
                      <Tag color={ex.score > 0.8 ? "green" : ex.score > 0.4 ? "orange" : "red"}>
                        {(ex.score * 100).toFixed(0)}%
                      </Tag>
                    </div>
                  ))}
                </div>
              )}

              {/* RAG vs Fine-tuning comparison */}
              {LESSON_SECTIONS[lessonStep].comparison && (
                <div className="comparison-section">
                  <Title level={5}>RAG vs Fine-tuning</Title>
                  <div className="comparison-table">
                    <div className="comp-header">
                      <div className="comp-cell comp-aspect"></div>
                      <div className="comp-cell comp-rag"><Tag color="blue">RAG</Tag></div>
                      <div className="comp-cell comp-ft"><Tag color="purple">Fine-tuning</Tag></div>
                    </div>
                    {LESSON_SECTIONS[lessonStep].comparison.map((row, i) => (
                      <div key={i} className="comp-row">
                        <div className="comp-cell comp-aspect"><Text strong>{row.aspect}</Text></div>
                        <div className="comp-cell comp-rag"><Text>{row.rag}</Text></div>
                        <div className="comp-cell comp-ft"><Text>{row.finetune}</Text></div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </Card>

            <div className="lesson-nav-btns">
              <Button disabled={lessonStep === 0} onClick={() => setLessonStep(lessonStep - 1)}>
                Previous
              </Button>
              {lessonStep < LESSON_SECTIONS.length - 1 ? (
                <Button type="primary" onClick={() => setLessonStep(lessonStep + 1)}>
                  Next <ArrowRightOutlined />
                </Button>
              ) : (
                <Button type="primary" icon={<ExperimentOutlined />} onClick={() => setMode("experiment")}>
                  Try It: Run a RAG Query
                </Button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ═══ EXPERIMENT MODE ═══ */}
      {mode === "experiment" && (
        <div className="rag-experiment">
          <Alert
            type="info"
            showIcon
            icon={<ExperimentOutlined />}
            message="Live RAG Pipeline"
            description="Type a question and watch each RAG step execute in real time. Try: 'How do I return a product?', 'How fast is shipping?', 'What's my warranty?'"
            style={{ marginBottom: 20 }}
          />

          {/* Query input */}
          <Card bordered={false} className="exp-query-card">
            <Title level={5}><SearchOutlined /> Your Question</Title>
            <Space.Compact style={{ width: "100%" }}>
              <Input
                placeholder="Ask something... (try: 'how do I get a refund?')"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onPressEnter={runRAG}
                size="large"
                disabled={ragStage > 0 && ragStage < 5}
              />
              <Button
                type="primary"
                size="large"
                icon={<ThunderboltOutlined />}
                onClick={runRAG}
                disabled={!query.trim() || (ragStage > 0 && ragStage < 5)}
              >
                Run RAG
              </Button>
            </Space.Compact>
            {ragStage >= 1 && (
              <Button type="link" icon={<ReloadOutlined />} onClick={resetExperiment} style={{ marginTop: 8 }}>
                Reset
              </Button>
            )}
          </Card>

          {/* Pipeline progress */}
          {ragStage >= 1 && (
            <Card bordered={false} className="exp-pipeline-card">
              <Steps
                current={ragStage - 1}
                size="small"
                items={[
                  { title: "Embed", icon: ragStage > 1 ? <CheckCircleOutlined style={{ color: "#34d399" }} /> : undefined },
                  { title: "Search", icon: ragStage > 2 ? <CheckCircleOutlined style={{ color: "#34d399" }} /> : undefined },
                  { title: "Augment", icon: ragStage > 3 ? <CheckCircleOutlined style={{ color: "#34d399" }} /> : undefined },
                  { title: "Generate", icon: ragStage > 4 ? <CheckCircleOutlined style={{ color: "#34d399" }} /> : undefined },
                ]}
              />
            </Card>
          )}

          {/* Stage 1: Embedding */}
          {ragStage >= 1 && queryEmbedding && (
            <Card bordered={false} className="exp-stage-card">
              <Title level={5}>
                <Tag color="blue">Step 1</Tag> Query Embedding
              </Title>
              <Paragraph type="secondary">Your question converted to a vector:</Paragraph>
              <div className="embedding-display">
                <code>"{query}" → [{queryEmbedding.map(v => v.toFixed(3)).join(", ")}]</code>
              </div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                (Simplified to 4 dimensions for demo. Real embeddings are 1536-4096 dimensions)
              </Text>
            </Card>
          )}

          {/* Stage 2: Search results */}
          {ragStage >= 2 && searchResults.length > 0 && (
            <Card bordered={false} className="exp-stage-card">
              <Title level={5}>
                <Tag color="purple">Step 2</Tag> Vector Search Results (Top 3)
              </Title>
              <Paragraph type="secondary">Documents ranked by cosine similarity to your query:</Paragraph>
              <div className="search-results">
                {searchResults.map((doc, i) => (
                  <div key={doc.id} className={`search-result-item ${i === 0 ? "top-result" : ""}`}>
                    <div className="sr-header">
                      <Space>
                        <Tag color={doc.similarity > 0.8 ? "green" : doc.similarity > 0.6 ? "orange" : "red"}>
                          {(doc.similarity * 100).toFixed(1)}% match
                        </Tag>
                        <Text strong>{doc.title}</Text>
                      </Space>
                      {i === 0 && <Tag color="gold">Best match</Tag>}
                    </div>
                    <Text type="secondary" style={{ fontSize: 13 }}>{doc.content}</Text>
                    <div className="sr-embedding">
                      <Text type="secondary" style={{ fontSize: 11 }}>
                        Vector: [{doc.embedding.map(v => v.toFixed(2)).join(", ")}]
                      </Text>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Stage 3: Augmented prompt */}
          {ragStage >= 3 && augmentedPrompt && (
            <Card bordered={false} className="exp-stage-card">
              <Title level={5}>
                <Tag color="orange">Step 3</Tag> Augmented Prompt (sent to LLM)
              </Title>
              <Paragraph type="secondary">Retrieved docs are stuffed into the prompt context:</Paragraph>
              <div className="augmented-prompt-display">
                <pre>{augmentedPrompt}</pre>
              </div>
            </Card>
          )}

          {/* Stage 4/5: Generated answer */}
          {ragStage >= 4 && (
            <Card bordered={false} className="exp-stage-card exp-answer-card">
              <Title level={5}>
                <Tag color="green">Step 4</Tag> LLM Generated Answer
              </Title>
              {ragStage === 4 && !generatedAnswer && (
                <div className="generating-indicator">
                  <RobotOutlined spin style={{ marginRight: 8 }} />
                  <Text type="secondary">Generating response...</Text>
                </div>
              )}
              {generatedAnswer && (
                <div className="generated-answer">
                  <RobotOutlined style={{ marginRight: 8, color: "#4a90e2" }} />
                  <Text>{generatedAnswer}</Text>
                </div>
              )}
            </Card>
          )}

          {/* Knowledge base reference */}
          <Card bordered={false} className="exp-kb-card">
            <Title level={5}><DatabaseOutlined /> Knowledge Base ({KNOWLEDGE_BASE.length} documents)</Title>
            <Paragraph type="secondary" style={{ fontSize: 13 }}>
              This is the "vector database" being searched. In production, this would be millions of documents.
            </Paragraph>
            <div className="kb-docs">
              {KNOWLEDGE_BASE.map(doc => (
                <div key={doc.id} className="kb-doc-item">
                  <FileTextOutlined style={{ color: "#64748b" }} />
                  <Text style={{ fontSize: 13 }}>{doc.title}</Text>
                </div>
              ))}
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

      {/* ═══ QUIZ MODE ═══ */}
      {mode === "quiz" && (
        <div className="rag-quiz">
          <Alert
            type="info"
            showIcon
            icon={<BulbOutlined />}
            message="Answer 5 questions about RAG. Score 3/5 or higher to complete this module."
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

      {/* ═══ RESULT MODE ═══ */}
      {mode === "result" && (
        <div className="rag-result">
          <Result
            status={passed ? "success" : "warning"}
            icon={passed ? <TrophyOutlined /> : undefined}
            title={passed ? "RAG Module Complete!" : "Not quite — try again"}
            subTitle={`You scored ${quizScore}/5 ${passed ? "— you understand how RAG gives LLMs superpowers!" : "— need at least 3/5"}`}
            extra={
              <Space>
                {passed && (
                  <Button type="primary" icon={<TrophyOutlined />} onClick={() => setMode("learn")}>
                    Review or Explore More
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
