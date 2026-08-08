import { useState } from "react";
import "./StepViewer.css";

/**
 * Reusable step-by-step trace viewer.
 * Shows the C++ execution trace with step navigation, progress bar, and details.
 */
export default function StepViewer({ steps, currentStep, onStepChange, title }) {
  const [expanded, setExpanded] = useState(true);

  if (!steps || steps.length === 0) return null;

  const step = steps[currentStep];
  const progress = ((currentStep + 1) / steps.length) * 100;

  return (
    <div className="step-viewer animate-in">
      {/* Header */}
      <div className="sv-header">
        <div className="sv-title-area">
          <h3 className="sv-title">{title || step.component}</h3>
          <span className="sv-step-count">
            Step {currentStep + 1} of {steps.length}
          </span>
        </div>
        <div className="sv-controls">
          <button
            className="sv-btn"
            onClick={() => onStepChange(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
          >
            ← Prev
          </button>
          <button
            className="sv-btn"
            onClick={() => onStepChange(Math.min(steps.length - 1, currentStep + 1))}
            disabled={currentStep === steps.length - 1}
          >
            Next →
          </button>
          <button
            className="sv-btn sv-btn-accent"
            onClick={() => onStepChange(steps.length - 1)}
          >
            Skip to End
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="sv-progress-bar">
        <div className="sv-progress-fill" style={{ width: `${progress}%` }} />
      </div>

      {/* Step dots */}
      <div className="sv-dots">
        {steps.map((s, i) => (
          <button
            key={i}
            className={`sv-dot ${i === currentStep ? "active" : ""} ${i < currentStep ? "done" : ""}`}
            onClick={() => onStepChange(i)}
            title={s.title}
          />
        ))}
      </div>

      {/* Current Step Content */}
      <div className="sv-content">
        <div className="sv-step-header">
          <span className="sv-component-badge">{step.component}</span>
          <h4 className="sv-step-title">{step.title}</h4>
        </div>

        {step.detail && (
          <p className="sv-step-detail">{step.detail}</p>
        )}

        {step.internal && step.internal.length > 0 && (
          <div className="sv-internal">
            <button
              className="sv-expand-btn"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? "▼" : "▶"} Math Details ({step.internal.length} lines)
            </button>
            {expanded && (
              <div className="sv-internal-content">
                {step.internal.map((line, i) => (
                  <div
                    key={i}
                    className={`sv-line ${
                      line === "" ? "sv-spacer" :
                      line.startsWith("  ") ? "sv-indented" :
                      line.includes("✓") || line.includes("✗") ? "sv-check" :
                      line.includes("→") ? "sv-arrow-line" :
                      ""
                    }`}
                  >
                    {line || "\u00A0"}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
