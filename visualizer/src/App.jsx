import { useState, useEffect } from "react";
import { Layout, Menu, Breadcrumb, Badge, Typography } from "antd";
import {
  HomeOutlined,
  ThunderboltOutlined,
  MessageOutlined,
  EyeOutlined,
  EditOutlined,
  FunctionOutlined,
  ApartmentOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  NodeIndexOutlined,
  CompressOutlined,
  ClusterOutlined,
  AppstoreOutlined,
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
    key: "ai",
    icon: <MessageOutlined />,
    label: "How AI Thinks",
    children: [
      { key: "text", icon: <MessageOutlined />, label: "Text Pipeline" },
      { key: "image", icon: <EyeOutlined />, label: "Vision Pipeline" },
      { key: "mnist", icon: <EditOutlined />, label: "Live Inference" },
      { key: "math", icon: <FunctionOutlined />, label: "Math Lab" },
    ],
  },
  {
    key: "cpp",
    icon: <ThunderboltOutlined />,
    label: "How AI Runs Fast",
    children: [
      { key: "arch", icon: <ApartmentOutlined />, label: "Architecture" },
      { key: "kernels", icon: <DashboardOutlined />, label: "SIMD Kernels" },
      { key: "memory", icon: <DatabaseOutlined />, label: "Memory Arena" },
      { key: "graph", icon: <NodeIndexOutlined />, label: "Graph Compiler" },
      { key: "quantize", icon: <CompressOutlined />, label: "Quantization" },
      { key: "threads", icon: <ClusterOutlined />, label: "Thread Pool" },
      { key: "tensor", icon: <AppstoreOutlined />, label: "Tensor Engine" },
    ],
  },
];

// Pages that are part of the guided learning path
const GUIDED_PAGES = ["math", "text", "image", "mnist"];

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
  const [visitedPages, setVisitedPages] = useState(() => {
    // Persist progress in localStorage
    try {
      const saved = localStorage.getItem("inferx-visited");
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });

  useEffect(() => {
    initNNWasm().then(() => setNNReady(true)).catch(() => setNNReady(true));
  }, []);

  // Track visited pages
  useEffect(() => {
    if (page !== "home" && !visitedPages.includes(page)) {
      const updated = [...visitedPages, page];
      setVisitedPages(updated);
      try { localStorage.setItem("inferx-visited", JSON.stringify(updated)); } catch {}
    }
  }, [page]);

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
        {page === "home" && <Landing onNavigate={navigate} visitedPages={visitedPages} />}
        {page === "arch" && <ArchExplorer />}
        {page === "text" && <TextPipeline />}
        {page === "image" && <ImagePipeline />}
        {page === "mnist" && <MNISTLive />}
        {page === "graph" && <GraphOptimizer />}
        {page === "memory" && <MemoryViz />}
        {page === "kernels" && <KernelViz />}
        {page === "quantize" && <QuantizeViz />}
        {page === "threads" && <ThreadPoolViz />}
        {page === "math" && <MathExplorer />}
        {page === "tensor" && <TensorPage />}
      </Content>

      {/* ═══ Guided Track (only on learning-path pages) ═══ */}
      {isGuidedPage && (
        <GuidedTrack
          currentPage={page}
          visitedPages={visitedPages}
          onNavigate={navigate}
        />
      )}

      <FeedbackWidget />
      <Footer />
    </Layout>
  );
}
