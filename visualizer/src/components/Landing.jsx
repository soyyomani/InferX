import { Typography, Button, Card, Row, Col, Statistic, Space, Divider, Steps, Progress, Collapse, Tag, Tooltip } from "antd";
import {
  RocketOutlined,
  ExperimentOutlined,
  MessageOutlined,
  EyeOutlined,
  EditOutlined,
  FunctionOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  NodeIndexOutlined,
  CompressOutlined,
  ClusterOutlined,
  AppstoreOutlined,
  PlayCircleOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ArrowRightOutlined,
  BookOutlined,
  TrophyOutlined,
  LockOutlined,
} from "@ant-design/icons";
import "./Landing.css";

const { Title, Paragraph, Text } = Typography;

// The 4-step guided learning path for students
const LEARNING_PATH = [
  {
    key: "math",
    title: "Math Lab",
    subtitle: "The math behind AI",
    desc: "Matrix multiplication, softmax, ReLU — interactive step-by-step calculations. No calculus needed, just curiosity. This is the foundation everything else builds on.",
    icon: <FunctionOutlined />,
    time: "8 min",
    prereq: null,
  },
  {
    key: "text",
    title: "Text Pipeline",
    subtitle: "How AI reads words",
    desc: "See how a sentence becomes numbers → embeddings → attention → prediction. The foundation of ChatGPT and every language model.",
    icon: <MessageOutlined />,
    time: "5 min",
    prereq: "Math Lab",
  },
  {
    key: "image",
    title: "Vision Pipeline",
    subtitle: "How AI sees images",
    desc: "Upload an image and watch a CNN process it: convolution → ReLU → pooling → classification. Every calculation shown.",
    icon: <EyeOutlined />,
    time: "6 min",
    prereq: "Text Pipeline",
  },
  {
    key: "mnist",
    title: "Live Inference",
    subtitle: "Draw and predict",
    desc: "Draw a digit with your mouse and watch the neural network activate in real time. See neurons fire layer by layer.",
    icon: <EditOutlined />,
    time: "3 min",
    prereq: "Vision Pipeline",
  },
];

// Advanced systems section
const SYSTEMS = [
  { key: "arch", icon: <RocketOutlined />, title: "Architecture", desc: "Full system overview" },
  { key: "kernels", icon: <DashboardOutlined />, title: "SIMD Kernels", desc: "ARM NEON, 22 GFLOPS" },
  { key: "memory", icon: <DatabaseOutlined />, title: "Memory Arena", desc: "918× faster than malloc" },
  { key: "graph", icon: <NodeIndexOutlined />, title: "Graph Compiler", desc: "Operator fusion passes" },
  { key: "quantize", icon: <CompressOutlined />, title: "Quantization", desc: "INT8: 4× compression" },
  { key: "threads", icon: <ClusterOutlined />, title: "Thread Pool", desc: "Parallel task execution" },
  { key: "tensor", icon: <AppstoreOutlined />, title: "Tensor Engine", desc: "Zero-copy ops" },
];

export default function Landing({ onNavigate, visitedPages = [] }) {
  const completedSteps = LEARNING_PATH.filter(s => visitedPages.includes(s.key)).length;
  const progressPercent = Math.round((completedSteps / LEARNING_PATH.length) * 100);

  // Find the next uncompleted step
  const nextStep = LEARNING_PATH.find(s => !visitedPages.includes(s.key));

  return (
    <div className="landing">
      {/* ═══ Hero Section ═══ */}
      <section className="landing-hero">
        <div className="hero-glow" />
        <div className="hero-content">
          <Tag color="blue" className="hero-tag">Interactive AI Education</Tag>
          <Title level={1} className="hero-title">
            Understand AI by <span className="hero-highlight">watching it think.</span>
          </Title>
          <Paragraph className="hero-desc">
            Not another textbook. Not another video. This is a live, interactive playground
            where you see every calculation — from raw text to prediction — happening in real time.
          </Paragraph>
          <Space size="large" className="hero-actions">
            <Button
              type="primary"
              size="large"
              icon={<PlayCircleOutlined />}
              className="hero-btn-primary"
              onClick={() => onNavigate(nextStep?.key || "text")}
            >
              {completedSteps === 0 ? "Start Learning" : "Continue Learning"}
            </Button>
            <Button
              size="large"
              icon={<ExperimentOutlined />}
              onClick={() => onNavigate("mnist")}
            >
              Jump to demo
            </Button>
          </Space>
        </div>

        {/* Floating stats */}
        <div className="hero-stats">
          <div className="hero-stat-pill">
            <ThunderboltOutlined /> Real C++ engine running in your browser
          </div>
          <div className="hero-stat-pill">
            <CheckCircleOutlined /> No signup required
          </div>
          <div className="hero-stat-pill">
            <BookOutlined /> Beginner friendly
          </div>
        </div>
      </section>

      {/* ═══ Learning Progress ═══ */}
      {completedSteps > 0 && (
        <section className="landing-progress">
          <Card className="progress-card" bordered={false}>
            <Row align="middle" gutter={16}>
              <Col flex="auto">
                <Space>
                  <TrophyOutlined style={{ fontSize: 20, color: "#fbbf24" }} />
                  <Text strong>Your Progress</Text>
                  <Text type="secondary">— {completedSteps} of {LEARNING_PATH.length} concepts explored</Text>
                </Space>
              </Col>
              <Col>
                <Progress
                  percent={progressPercent}
                  steps={4}
                  size="small"
                  strokeColor="#4a90e2"
                />
              </Col>
            </Row>
          </Card>
        </section>
      )}

      {/* ═══ Guided Learning Path ═══ */}
      <section className="landing-path">
        <div className="section-header">
          <Title level={3}>
            <BookOutlined style={{ marginRight: 8 }} />
            Start Here — 4 Steps to Understand AI
          </Title>
          <Paragraph type="secondary">
            Follow this path in order. Each concept builds on the last. Takes about 20 minutes total.
          </Paragraph>
        </div>

        <div className="path-grid">
          {LEARNING_PATH.map((step, idx) => {
            const isCompleted = visitedPages.includes(step.key);
            const isNext = step.key === nextStep?.key;
            const isLocked = idx > 0 && !visitedPages.includes(LEARNING_PATH[idx - 1].key) && !isCompleted;

            return (
              <div
                key={step.key}
                className={`path-card ${isCompleted ? "completed" : ""} ${isNext ? "next" : ""} ${isLocked ? "locked" : ""}`}
                onClick={() => !isLocked && onNavigate(step.key)}
              >
                <div className="path-card-number">
                  {isCompleted ? <CheckCircleOutlined /> : <span>{idx + 1}</span>}
                </div>
                <div className="path-card-icon">{step.icon}</div>
                <div className="path-card-body">
                  <Text strong className="path-card-title">{step.title}</Text>
                  <Text type="secondary" className="path-card-subtitle">{step.subtitle}</Text>
                  <Paragraph type="secondary" className="path-card-desc">{step.desc}</Paragraph>
                </div>
                <div className="path-card-footer">
                  <Tag>{step.time}</Tag>
                  {isCompleted && <Tag color="success">Completed</Tag>}
                  {isNext && <Tag color="blue">Up Next</Tag>}
                  {isLocked && (
                    <Tooltip title={`Complete "${step.prereq}" first`}>
                      <Tag icon={<LockOutlined />}>Locked</Tag>
                    </Tooltip>
                  )}
                </div>
                {isNext && (
                  <Button
                    type="primary"
                    icon={<ArrowRightOutlined />}
                    className="path-card-cta"
                    onClick={(e) => { e.stopPropagation(); onNavigate(step.key); }}
                  >
                    Start
                  </Button>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* ═══ Stats Bar ═══ */}
      <section className="landing-numbers">
        <Row gutter={[24, 24]} justify="center">
          <Col><div className="stat-glass"><Statistic title="GFLOPS" value={22} valueStyle={{ color: "#4a90e2" }} /></div></Col>
          <Col><div className="stat-glass"><Statistic title="vs malloc" value="918x" valueStyle={{ color: "#34d399" }} /></div></Col>
          <Col><div className="stat-glass"><Statistic title="Tests" value="216+" valueStyle={{ color: "#a78bfa" }} /></div></Col>
          <Col><div className="stat-glass"><Statistic title="Accuracy" value="100%" valueStyle={{ color: "#fbbf24" }} /></div></Col>
          <Col><div className="stat-glass"><Statistic title="Standard" value="C++20" valueStyle={{ color: "#fb923c" }} /></div></Col>
        </Row>
      </section>

      {/* ═══ How It All Connects ═══ */}
      <section className="landing-connect">
        <Card bordered={false} className="connect-card">
          <Paragraph style={{ fontSize: 16, textAlign: "center", marginBottom: 8 }}>
            <Text code>Attention(Q, K, V)</Text> is an algorithm.{" "}
            Making it run at <Text strong>7,485 images/sec</Text> with{" "}
            <Text strong>552 bytes</Text> peak memory is systems engineering.
          </Paragraph>
          <Paragraph type="secondary" style={{ textAlign: "center", marginBottom: 0 }}>
            This project teaches you both.
          </Paragraph>
        </Card>
      </section>

      {/* ═══ Advanced: Systems (Collapsed) ═══ */}
      <section className="landing-advanced">
        <Collapse
          ghost
          items={[{
            key: "systems",
            label: (
              <Space>
                <ThunderboltOutlined />
                <Text strong>Advanced: How the C++ Engine Works</Text>
                <Tag>7 deep-dives</Tag>
              </Space>
            ),
            children: (
              <Row gutter={[12, 12]}>
                {SYSTEMS.map(s => (
                  <Col xs={24} sm={12} md={8} key={s.key}>
                    <Card
                      hoverable
                      size="small"
                      onClick={() => onNavigate(s.key)}
                      className="system-card"
                      styles={{ body: { padding: "14px 16px" } }}
                    >
                      <Space>
                        <span style={{ fontSize: 18, color: "#4a90e2" }}>{s.icon}</span>
                        <div>
                          <Text strong>{s.title}</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: 12 }}>{s.desc}</Text>
                        </div>
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            ),
          }]}
        />
      </section>

      {/* ═══ Footer ═══ */}
      <div className="landing-footer">
        <Space split={<Divider type="vertical" />} size="small" wrap>
          <Text type="secondary">Open source</Text>
          <Text type="secondary">MIT license</Text>
          <Text type="secondary">No external ML deps</Text>
          <Text type="secondary">Apple Silicon optimized</Text>
        </Space>
      </div>
    </div>
  );
}
