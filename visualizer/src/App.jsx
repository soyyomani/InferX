import { useState } from "react";
import PipelineView from "./components/PipelineView";
import InteractiveBuilder from "./components/InteractiveBuilder";
import "./App.css";

export default function App() {
  const [tab, setTab] = useState("interactive");

  return (
    <div className="app">
      <header>
        <h1>InferX Tensor Visualizer</h1>
        <p>Actual C++ tensor engine compiled to WebAssembly — see every internal step</p>
      </header>

      <nav className="tabs">
        <button className={tab === "interactive" ? "active" : ""} onClick={() => setTab("interactive")}>
          Build Your Own
        </button>
        <button className={tab === "presets" ? "active" : ""} onClick={() => setTab("presets")}>
          Preset Examples
        </button>
      </nav>

      <main>
        {tab === "interactive" && <InteractiveBuilder />}
        {tab === "presets" && <PipelineView />}
      </main>

      <footer>
        <p>InferX — C++20 AI Inference Engine | Running real compiled C++ in your browser</p>
      </footer>
    </div>
  );
}
