import { useState, useEffect } from "react";
import TextPipeline from "./components/TextPipeline";
import ImagePipeline from "./components/ImagePipeline";
import MathExplorer from "./components/MathExplorer";
import TensorPage from "./components/TensorPage";
import Landing from "./components/Landing";
import Footer from "./components/Footer";
import FeedbackWidget from "./components/FeedbackWidget";
import { IconBolt, IconChat, IconImage, IconSigma, IconGrid, IconHome } from "./components/Icons";
import { initNNWasm, isNNReady } from "./engine/nn_wasm";
import "./App.css";

const PAGES = [
  { id: "home", label: "Home", Icon: IconHome },
  { id: "text", label: "Text AI", Icon: IconChat },
  { id: "image", label: "Vision AI", Icon: IconImage },
  { id: "math", label: "Math Lab", Icon: IconSigma },
  { id: "tensor", label: "Tensors", Icon: IconGrid },
];

export default function App() {
  const [page, setPage] = useState("home");
  const [nnReady, setNNReady] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    initNNWasm().then(() => setNNReady(true)).catch(() => setNNReady(true));
  }, []);

  return (
    <div className="app">
      <nav className="topnav">
        <div className="nav-brand" onClick={() => setPage("home")}>
          <IconBolt size={22} className="brand-icon-svg" />
          <span className="brand-text">InferX</span>
          <span className="brand-badge">Educational AI Engine</span>
        </div>

        <button className="mobile-menu-btn" onClick={() => setMenuOpen(!menuOpen)}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
        </button>

        <div className={`nav-links ${menuOpen ? "open" : ""}`}>
          {PAGES.map((p) => (
            <button
              key={p.id}
              className={`nav-link ${page === p.id ? "active" : ""}`}
              onClick={() => { setPage(p.id); setMenuOpen(false); }}
            >
              <span className="nav-icon"><p.Icon size={16} /></span>
              <span className="nav-label">{p.label}</span>
            </button>
          ))}
        </div>

        <div className="nav-status">
          <span className={`status-dot ${nnReady ? "ready" : "loading"}`} />
          <span className="status-text">{nnReady ? "Engine Ready" : "Loading..."}</span>
        </div>
      </nav>

      <main className="main-content">
        {page === "home" && <Landing onNavigate={setPage} />}
        {page === "text" && <TextPipeline />}
        {page === "image" && <ImagePipeline />}
        {page === "math" && <MathExplorer />}
        {page === "tensor" && <TensorPage />}
      </main>

      <FeedbackWidget />
      <Footer />
    </div>
  );
}
