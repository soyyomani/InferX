import { useState } from "react";
import "./FeedbackWidget.css";

export default function FeedbackWidget() {
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ name: "", email: "", feedback: "" });
  const [status, setStatus] = useState("idle"); // idle | sending | sent | error

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus("sending");

    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      if (res.ok) {
        setStatus("sent");
        setForm({ name: "", email: "", feedback: "" });
        setTimeout(() => {
          setStatus("idle");
          setOpen(false);
        }, 3000);
      } else {
        setStatus("error");
      }
    } catch {
      setStatus("error");
    }
  };

  return (
    <div className="feedback-widget">
      <div className="feedback-widget-inner">
        {/* Chat panel */}
        {open && (
          <div className="feedback-panel">
            <div className="feedback-header">
              <h3>Feedback</h3>
              <p>Help us improve InferX</p>
            </div>

            {status === "sent" ? (
              <div className="feedback-success">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="2" strokeLinecap="round">
                  <path d="M20 6L9 17l-5-5" />
                </svg>
                <p>Thanks for your feedback!</p>
              </div>
            ) : (
              <form className="feedback-form" onSubmit={handleSubmit}>
                <div className="feedback-field">
                  <label htmlFor="fb-name">Name</label>
                  <input
                    id="fb-name"
                    name="name"
                    type="text"
                    placeholder="Your name"
                    value={form.name}
                    onChange={handleChange}
                    required
                  />
                </div>

                <div className="feedback-field">
                  <label htmlFor="fb-email">Email</label>
                  <input
                    id="fb-email"
                    name="email"
                    type="email"
                    placeholder="you@example.com"
                    value={form.email}
                    onChange={handleChange}
                    required
                  />
                </div>

                <div className="feedback-field">
                  <label htmlFor="fb-feedback">What can we improve?</label>
                  <textarea
                    id="fb-feedback"
                    name="feedback"
                    placeholder="What wasn't clear? What would make this better?"
                    value={form.feedback}
                    onChange={handleChange}
                    rows={4}
                    required
                  />
                </div>

                <button
                  type="submit"
                  className="feedback-submit"
                  disabled={status === "sending"}
                >
                  {status === "sending" ? "Sending..." : "Send Feedback"}
                </button>

                {status === "error" && (
                  <p className="feedback-error">Something went wrong. Try again.</p>
                )}
              </form>
            )}
          </div>
        )}

        {/* Chat bubble toggle */}
        <button
          className={`feedback-toggle ${open ? "active" : ""}`}
          onClick={() => setOpen(!open)}
          aria-label={open ? "Close feedback" : "Send feedback"}
        >
          {open ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}
