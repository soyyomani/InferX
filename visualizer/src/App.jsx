import { useState } from "react";
import InteractiveBuilder from "./components/InteractiveBuilder";
import AIPipeline from "./components/AIPipeline";
import "./App.css";

export default function App() {
  const [tab, setTab] = useState("ai-pipeline");

  return (
    <div className="app">
      <header>
        <h1>InferX</h1>
        <p>C++20 AI Inference Engine | Running real compiled C++ in your browser</p>
      </header>

      <nav className="tabs">
        <button className={tab === "ai-pipeline" ? "active" : ""} onClick={() => setTab("ai-pipeline")}>
          How AI Thinks
        </button>
        <button className={tab === "interactive" ? "active" : ""} onClick={() => setTab("interactive")}>
          Tensor Playground
        </button>
      </nav>

      <main>
        {tab === "ai-pipeline" && <AIPipeline />}
        {tab === "interactive" && <InteractiveBuilder />}
      </main>

      <footer>
        <p>InferX — C++20 AI Inference Engine | Running real compiled C++ in your browser</p>
      </footer>
    </div>
  );
}
