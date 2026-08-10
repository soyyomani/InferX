import { useState, useEffect, useRef } from "react";
import TextPipeline from "./components/TextPipeline";
import ImagePipeline from "./components/ImagePipeline";
import MathExplorer from "./components/MathExplorer";
import TensorPage from "./components/TensorPage";
import Landing from "./components/Landing";
import Footer from "./components/Footer";
import FeedbackWidget from "./components/FeedbackWidget";
import { IconBolt, IconChat, IconImage, IconSigma, IconGrid, IconHome, IconPencil, IconNetwork, IconBrain, IconLayers } from "./components/Icons";
import { initNNWasm, isNNReady } from "./engine/nn_wasm";
import MNISTLive from "./components/MNISTLive";
import GraphOptimizer from "./components/GraphOptimizer";
import ArchExplorer from "./components/ArchExplorer";
import MemoryViz from "./components/MemoryViz";
import KernelViz from "./components/KernelViz";
import QuantizeViz from "./components/QuantizeViz";
import ThreadPoolViz from "./components/ThreadPoolViz";
import "./App.css";

/*
 * Navigation Architecture:
 * ────────────────────────
 * Instead of 12 flat items, we group into 4 clear categories:
 *
 *   [Home]  [AI Demos ▾]  [Engine Internals ▾]  [Playground ▾]
 *
 * Each dropdown reveals 2-4 items max. User always knows WHERE they are.
 * This is the "progressive disclosure" pattern — show only what's needed.
 */

const NAV_GROUPS = [
  {
    id: "ai",
    label: "How AI Thinks",
    items: [
      { id: "text", label: "Text Pipeline", desc: "Transformer: tokenize → attend → predict", Icon: IconChat },
      { id: "image", label: "Vision Pipeline", desc: "CNN: convolve → pool → classify", Icon: IconImage },
      { id: "mnist", label: "Live Inference", desc: "Draw a digit, get prediction", Icon: IconPencil },
      { id: "math", label: "Math Lab", desc: "MatMul, Softmax, ReLU step by step", Icon: IconSigma },
    ],
  },
  {
    id: "cpp",
    label: "How AI Runs Fast",
    items: [
      { id: "arch", label: "Architecture", desc: "Full system overview", Icon: IconLayers },
      { id: "kernels", label: "SIMD Kernels", desc: "NEON matmul, 22 GFLOPS", Icon: IconBolt },
      { id: "memory", label: "Memory Arena", desc: "918× faster than malloc", Icon: IconBrain },
      { id: "graph", label: "Graph Compiler", desc: "Operator fusion passes", Icon: IconNetwork },
      { id: "quantize", label: "Quantization", desc: "INT8: 4× compression", Icon: IconGrid },
      { id: "threads", label: "Thread Pool", desc: "Parallel task execution", Icon: IconSigma },
      { id: "tensor", label: "Tensor Engine", desc: "Zero-copy ops, broadcasting", Icon: IconGrid },
    ],
  },
];

// Flat lookup for rendering
const ALL_PAGES = NAV_GROUPS.flatMap(g => g.items);

export default function App() {
  const [page, setPage] = useState("home");
  const [nnReady, setNNReady] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [openDropdown, setOpenDropdown] = useState(null);
  const navRef = useRef(null);

  useEffect(() => {
    initNNWasm().then(() => setNNReady(true)).catch(() => setNNReady(true));
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (navRef.current && !navRef.current.contains(e.target)) {
        setOpenDropdown(null);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const navigate = (id) => {
    setPage(id);
    setOpenDropdown(null);
    setMenuOpen(false);
  };

  // Find which group the current page belongs to
  const currentGroup = NAV_GROUPS.find(g => g.items.some(i => i.id === page));
  const currentPage = ALL_PAGES.find(p => p.id === page);

  return (
    <div className="app">
      {/* ═══ Top Navigation ═══ */}
      <nav className="topnav" ref={navRef}>
        <div className="nav-brand" onClick={() => navigate("home")}>
          <IconBolt size={22} className="brand-icon-svg" />
          <span className="brand-text">InferX</span>
        </div>

        {/* Mobile menu toggle */}
        <button className="mobile-menu-btn" onClick={() => setMenuOpen(!menuOpen)}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/>
          </svg>
        </button>

        {/* Desktop: grouped dropdowns */}
        <div className={`nav-groups ${menuOpen ? "open" : ""}`}>
          {NAV_GROUPS.map((group) => (
            <div key={group.id} className="nav-group">
              <button
                className={`nav-group-btn ${currentGroup?.id === group.id ? "active" : ""} ${openDropdown === group.id ? "expanded" : ""}`}
                onClick={() => setOpenDropdown(openDropdown === group.id ? null : group.id)}
              >
                <span>{group.label}</span>
                <svg className="chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <polyline points="6 9 12 15 18 9"/>
                </svg>
              </button>

              {/* Dropdown */}
              {openDropdown === group.id && (
                <div className="nav-dropdown">
                  {group.items.map((item) => (
                    <button
                      key={item.id}
                      className={`dropdown-item ${page === item.id ? "active" : ""}`}
                      onClick={() => navigate(item.id)}
                    >
                      <span className="dropdown-icon"><item.Icon size={18} /></span>
                      <div className="dropdown-text">
                        <span className="dropdown-label">{item.label}</span>
                        <span className="dropdown-desc">{item.desc}</span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Status + current page indicator */}
        <div className="nav-right">
          {currentPage && page !== "home" && (
            <span className="nav-current">{currentPage.label}</span>
          )}
          <div className="nav-status">
            <span className={`status-dot ${nnReady ? "ready" : "loading"}`} />
          </div>
        </div>
      </nav>

      {/* ═══ Breadcrumb (shows context when deep in a section) ═══ */}
      {page !== "home" && currentGroup && (
        <div className="breadcrumb">
          <button onClick={() => navigate("home")}>Home</button>
          <span className="bc-sep">/</span>
          <span className="bc-group">{currentGroup.label}</span>
          <span className="bc-sep">/</span>
          <span className="bc-current">{currentPage?.label}</span>
        </div>
      )}

      {/* ═══ Main Content ═══ */}
      <main className="main-content">
        {page === "home" && <Landing onNavigate={navigate} />}
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
      </main>

      <FeedbackWidget />
      <Footer />
    </div>
  );
}
