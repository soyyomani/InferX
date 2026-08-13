import { Button, Steps, Space, Typography, Progress } from "antd";
import {
  ArrowLeftOutlined,
  ArrowRightOutlined,
  HomeOutlined,
  MessageOutlined,
  FunctionOutlined,
  EyeOutlined,
  EditOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  TrophyOutlined,
} from "@ant-design/icons";
import "./GuidedTrack.css";

const { Text } = Typography;

const LEARNING_PATH = [
  { key: "math", title: "Math Lab", icon: <FunctionOutlined /> },
  { key: "text", title: "Text Pipeline", icon: <MessageOutlined /> },
  { key: "image", title: "Vision Pipeline", icon: <EyeOutlined /> },
  { key: "mnist", title: "Live Inference", icon: <EditOutlined /> },
];

/**
 * GuidedTrack — shows at the bottom of learning path pages.
 * Provides Previous / Next navigation and current position indicator.
 */
export default function GuidedTrack({ currentPage, visitedPages = [], onNavigate }) {
  const currentIdx = LEARNING_PATH.findIndex(s => s.key === currentPage);

  // Not on a guided-path page — don't render
  if (currentIdx === -1) return null;

  const prevStep = currentIdx > 0 ? LEARNING_PATH[currentIdx - 1] : null;
  const nextStep = currentIdx < LEARNING_PATH.length - 1 ? LEARNING_PATH[currentIdx + 1] : null;
  const isLastStep = currentIdx === LEARNING_PATH.length - 1;
  const completedCount = LEARNING_PATH.filter(s => visitedPages.includes(s.key)).length;

  // Steps items for antd Steps
  const stepsItems = LEARNING_PATH.map((step, idx) => {
    const isCompleted = visitedPages.includes(step.key);
    const isCurrent = idx === currentIdx;
    return {
      title: step.title,
      icon: isCurrent
        ? step.icon
        : isCompleted
          ? <CheckCircleOutlined style={{ color: "#34d399" }} />
          : <ClockCircleOutlined style={{ color: "#fbbf24" }} />,
      status: isCurrent ? "process" : isCompleted ? "finish" : "wait",
    };
  });

  return (
    <div className="guided-track">
      <div className="guided-track-inner">
        {/* Progress steps */}
        <div className="guided-steps-row">
          <Steps
            current={currentIdx}
            items={stepsItems}
            size="small"
            className="guided-steps"
          />
        </div>

        {/* Navigation buttons */}
        <div className="guided-nav-row">
          <div className="guided-nav-left">
            {prevStep ? (
              <Button
                icon={<ArrowLeftOutlined />}
                onClick={() => onNavigate(prevStep.key)}
              >
                {prevStep.title}
              </Button>
            ) : (
              <Button
                icon={<HomeOutlined />}
                onClick={() => onNavigate("home")}
              >
                Home
              </Button>
            )}
          </div>

          <div className="guided-nav-center">
            <Text type="secondary" className="guided-position">
              Step {currentIdx + 1} of {LEARNING_PATH.length}
            </Text>
          </div>

          <div className="guided-nav-right">
            {nextStep ? (
              <Button
                type="primary"
                onClick={() => onNavigate(nextStep.key)}
              >
                Next: {nextStep.title} <ArrowRightOutlined />
              </Button>
            ) : isLastStep ? (
              <Button
                type="primary"
                icon={<TrophyOutlined />}
                onClick={() => onNavigate("home")}
                style={{ background: "#34d399", borderColor: "#34d399" }}
              >
                Complete!
              </Button>
            ) : null}
          </div>
        </div>

        {/* Completion celebration */}
        {completedCount === LEARNING_PATH.length && (
          <div className="guided-complete-banner">
            <TrophyOutlined style={{ fontSize: 18, color: "#fbbf24" }} />
            <Text strong style={{ color: "#fbbf24" }}>
              You've explored all 4 core concepts!
            </Text>
            <Text type="secondary"> Try the Advanced section for C++ engine internals.</Text>
          </div>
        )}
      </div>
    </div>
  );
}
