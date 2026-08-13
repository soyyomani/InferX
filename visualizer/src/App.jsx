import { useState, useEffect } from "react";
import { Layout, Menu, Breadcrumb, Badge, Typography, Result, Tag } from "antd";
import {
  HomeOutlined,
  ThunderboltOutlined,
  ApartmentOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  NodeIndexOutlined,
  CompressOutlined,
  ClusterOutlined,
  AppstoreOutlined,
  ClockCircleOutlined,
} from "@ant-design/icons";
import TextPipeline from "./components/TextPipeline";
import ImagePipeline from "./components/ImagePipeline";
import MathExplorer from "./components/MathExplorer";
import TensorPage from "./components/TensorPage";
import Landing from "./components/Landing";
import Footer from "./components/Footer";
import FeedbackWidget from "./components/FeedbackWidget";
import GuidedTrack from "./components/GuidedTrack";
import { initNNWasm, isNNReady } from "./engine/nn_wasm";
import MNISTLive from "./components/MNISTLive";
import GraphOptimizer from "./components/GraphOptimizer";
import ArchExplorer from "./components/ArchExplorer";
import MemoryViz from "./components/MemoryViz";
import KernelViz from "./components/KernelViz";
import QuantizeViz from "./components/QuantizeViz";
import ThreadPoolViz from "./components/ThreadPoolViz";
import "./App.css";

const { Header, Content } = Layout;

const NAV_ITEMS = [
  {
    key: "home",
    icon: <HomeOutlined />,
    label: "Home",
  },
  {
    key: "cpp",
    icon: <ThunderboltOutlined />,
    label: "How AI Runs Fast (Coming Soon)",
    children: [
      { key: "arch", icon: <ApartmentOutlined />, label: "Architecture", disabled: true },
      { key: "kernels", icon: <DashboardOutlined />, label: "SIMD Kernels", disabled: true },
      { key: "memory", icon: <DatabaseOutlined />, label: "Memory Arena", disabled: true },
      { key: "graph", icon: <NodeIndexOutlined />, label: "Graph Compiler", disabled: true },
      { key: "quantize", icon: <CompressOutlined />, label: "Quantization", disabled: true },
      { key: "threads", icon: <ClusterOutlined />, label: "Thread Pool", disabled: true },
      { key: "tensor", icon: <AppstoreOutlined />, label: "Tensor Engine", disabled: true },
    ],
  },
];

// Pages that are part of the guided learning path
const GUIDED_PAGES = ["math", "text", "image", "mnist"];

// Pages that are not yet available (show "Coming Soon" banner)
const UPCOMING_PAGES = ["arch", "kernels", "memory", "graph", "quantize", "threads", "tensor"];

function ComingSoon({ title }) {
  return (
    <Result
      icon={<ClockCircleOutlined style={{ color: "#fbbf24" }} />}
      title={<span>{title} <Tag color="gold">Coming Soon</Tag></span>}
      subTitle="This module is currently under development. Check back soon for interactive deep-dives into C++ engine internals."
      style={{ marginTop: 60 }}
    />
  );
}

const PAGE_LABELS = {
  home: "Home",
  text: "Text Pipeline",
  image: "Vision Pipeline",
  mnist: "Live Inference",
  math: "Math Lab",
  arch: "Architecture",
  kernels: "SIMD Kernels",
  memory: "Memory Arena",
  graph: "Graph Compiler",
  quantize: "Quantization",
  threads: "Thread Pool",
  tensor: "Tensor Engine",
};

const GROUP_LABELS = {
  ai: "How AI Thinks",
  cpp: "How AI Runs Fast",
};

function getGroupForPage(pageKey) {
  const aiPages = ["text", "image", "mnist", "math"];
  const cppPages = ["arch", "kernels", "memory", "graph", "quantize", "threads", "tensor"];
  if (aiPages.includes(pageKey)) return "ai";
  if (cppPages.includes(pageKey)) return "cpp";
  return null;
}

export default function App() {
  const [page, setPage] = useState("home");
  const [nnReady, setNNReady] = useState(false);

  // completedPages = pages the user has ACTUALLY finished (not just visited)
  const [completedPages, setCompletedPages] = useState(() => {
    try {
      const saved = localStorage.getItem("inferx-completed");
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });

  useEffect(() => {
    initNNWasm().then(() => setNNReady(true)).catch(() => setNNReady(true));
  }, []);

  // Called by child components when the user finishes a module
  const markComplete = (pageKey) => {
    if (!completedPages.includes(pageKey)) {
      const updated = [...completedPages, pageKey];
      setCompletedPages(updated);
      try { localStorage.setItem("inferx-completed", JSON.stringify(updated)); } catch {}
    }
  };

  const navigate = (key) => {
    if (key === "ai" || key === "cpp") return;
    setPage(key);
    // Scroll to top on page change
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleMenuClick = ({ key }) => navigate(key);

  const currentGroup = getGroupForPage(page);
  const isGuidedPage = GUIDED_PAGES.includes(page);

  // Breadcrumb
  const breadcrumbItems = [
    { title: <a onClick={() => navigate("home")}>Home</a> },
  ];
  if (page !== "home" && currentGroup) {
    breadcrumbItems.push({ title: GROUP_LABELS[currentGroup] });
    breadcrumbItems.push({ title: PAGE_LABELS[page] });
  }

  const selectedKeys = page === "home" ? ["home"] : [page];
  const openKeys = currentGroup ? [currentGroup] : [];

  return (
    <Layout className="app-layout">
      {/* ═══ Header ═══ */}
      <Header className="app-header">
        <div className="nav-brand" onClick={() => navigate("home")}>
          <ThunderboltOutlined className="brand-icon-svg" />
          <span className="brand-text">InferX</span>
        </div>

        <Menu
          mode="horizontal"
          theme="dark"
          items={NAV_ITEMS}
          selectedKeys={selectedKeys}
          defaultOpenKeys={openKeys}
          onClick={handleMenuClick}
          className="app-menu"
          disabledOverflow
        />

        <div className="nav-status-area">
          <Badge
            status={nnReady ? "success" : "processing"}
            text={
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                {nnReady ? "Engine Ready" : "Loading..."}
              </Typography.Text>
            }
          />
        </div>
      </Header>

      {/* ═══ Breadcrumb ═══ */}
      {page !== "home" && (
        <div className="app-breadcrumb">
          <Breadcrumb items={breadcrumbItems} />
        </div>
      )}

      {/* ═══ Main Content ═══ */}
      <Content className="main-content">
        {page === "home" && <Landing onNavigate={navigate} visitedPages={completedPages} />}
        {page === "text" && <TextPipeline onComplete={() => markComplete("text")} />}
        {page === "image" && <ImagePipeline onComplete={() => markComplete("image")} />}
        {page === "mnist" && <MNISTLive onComplete={() => markComplete("mnist")} />}
        {page === "math" && <MathExplorer onComplete={() => markComplete("math")} />}
        {/* "How AI Runs Fast" pages — show Coming Soon banner */}
        {UPCOMING_PAGES.includes(page) && <ComingSoon title={PAGE_LABELS[page]} />}
      </Content>

      {/* ═══ Guided Track (only on learning-path pages) ═══ */}
      {isGuidedPage && (
        <GuidedTrack
          currentPage={page}
          visitedPages={completedPages}
          onNavigate={navigate}
        />
      )}

      <FeedbackWidget />
      <Footer />
    </Layout>
  );
}
