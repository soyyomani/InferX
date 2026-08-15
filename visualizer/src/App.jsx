import { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from "react-router-dom";
import { Layout, Breadcrumb, Badge, Typography, Result, Tag } from "antd";
import {
  ClockCircleOutlined,
} from "@ant-design/icons";
import MathExplorer from "./components/MathExplorer";
import TextPipeline from "./components/TextPipeline";
import ImagePipeline from "./components/ImagePipeline";
import MNISTLive from "./components/MNISTLive";
import Landing from "./components/Landing";
import Footer from "./components/Footer";
import FeedbackWidget from "./components/FeedbackWidget";
import GuidedTrack from "./components/GuidedTrack";
import { initNNWasm } from "./engine/nn_wasm";
import "./App.css";

const { Header, Content } = Layout;

// Sequential learning path
const GUIDED_PAGES = ["math", "textai", "visionai", "mnist"];

const PAGE_LABELS = {
  math: "Math Lab",
  textai: "How Text AI Works",
  visionai: "How Vision AI Works",
  mnist: "MNIST Live",
};

const GROUP_LABELS = {
  ai: "Learn AI",
};

function getGroupForPage(pageKey) {
  if (GUIDED_PAGES.includes(pageKey)) return "ai";
  return null;
}

function AppShell() {
  const navigate = useNavigate();
  const location = useLocation();
  const [nnReady, setNNReady] = useState(false);

  const [completedPages, setCompletedPages] = useState(() => {
    try {
      const saved = localStorage.getItem("inferx-completed");
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });

  useEffect(() => {
    initNNWasm().then(() => setNNReady(true)).catch(() => setNNReady(true));
  }, []);

  const markComplete = (pageKey) => {
    if (!completedPages.includes(pageKey)) {
      const updated = [...completedPages, pageKey];
      setCompletedPages(updated);
      try { localStorage.setItem("inferx-completed", JSON.stringify(updated)); } catch {}
    }
  };

  const goTo = (key) => {
    // Enforce sequential order: can only access a page if previous is completed
    if (GUIDED_PAGES.includes(key)) {
      const idx = GUIDED_PAGES.indexOf(key);
      if (idx > 0 && !completedPages.includes(GUIDED_PAGES[idx - 1])) {
        // Redirect to the first incomplete page
        const firstIncomplete = GUIDED_PAGES.find(p => !completedPages.includes(p)) || GUIDED_PAGES[0];
        const path = `/${firstIncomplete}`;
        navigate(path);
        window.scrollTo({ top: 0, behavior: "smooth" });
        return;
      }
    }
    const path = key === "home" ? "/" : `/${key}`;
    navigate(path);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  // Route guard: if user lands on a locked route (e.g. via URL), redirect
  useEffect(() => {
    const page = location.pathname.replace("/", "");
    if (GUIDED_PAGES.includes(page)) {
      const idx = GUIDED_PAGES.indexOf(page);
      if (idx > 0 && !completedPages.includes(GUIDED_PAGES[idx - 1])) {
        const firstIncomplete = GUIDED_PAGES.find(p => !completedPages.includes(p)) || GUIDED_PAGES[0];
        navigate(`/${firstIncomplete}`, { replace: true });
      }
    }
  }, [location.pathname, completedPages, navigate]);

  // Determine current page from URL
  const currentPage = location.pathname.replace("/", "") || "home";
  const currentGroup = getGroupForPage(currentPage);
  const isGuidedPage = GUIDED_PAGES.includes(currentPage);

  // Breadcrumb
  const breadcrumbItems = [
    { title: <a onClick={() => goTo("home")}>Home</a> },
  ];
  if (currentPage !== "home" && currentGroup) {
    breadcrumbItems.push({ title: GROUP_LABELS[currentGroup] });
    breadcrumbItems.push({ title: PAGE_LABELS[currentPage] });
  }

  return (
    <Layout className="app-layout">
      {/* ═══ Header ═══ */}
      <Header className="app-header">
        <div className="nav-brand" onClick={() => goTo("home")}>
          <img src="/logo.svg" alt="InferX" className="brand-logo" />
          <span className="brand-text">InferX</span>
        </div>

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
      {currentPage !== "home" && (
        <div className="app-breadcrumb">
          <Breadcrumb items={breadcrumbItems} />
        </div>
      )}

      {/* ═══ Main Content ═══ */}
      <Content className="main-content">
        <Routes>
          <Route path="/" element={<Landing onNavigate={goTo} visitedPages={completedPages} />} />
          <Route path="/math" element={<MathExplorer onComplete={() => markComplete("math")} />} />
          <Route path="/textai" element={<TextPipeline onComplete={() => markComplete("textai")} />} />
          <Route path="/visionai" element={<ImagePipeline onComplete={() => markComplete("visionai")} />} />
          <Route path="/mnist" element={<MNISTLive onComplete={() => markComplete("mnist")} />} />
          <Route path="*" element={<Landing onNavigate={goTo} visitedPages={completedPages} />} />
        </Routes>
      </Content>

      {/* ═══ Guided Track ═══ */}
      {isGuidedPage && (
        <GuidedTrack
          currentPage={currentPage}
          visitedPages={completedPages}
          onNavigate={goTo}
        />
      )}

      <FeedbackWidget />
      <Footer />
    </Layout>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  );
}
