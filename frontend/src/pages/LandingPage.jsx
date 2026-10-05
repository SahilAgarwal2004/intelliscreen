import React, { useState } from "react";
import { Shield, Sparkles, Eye, Mic, AlertTriangle, ArrowRight, CheckCircle2, Cpu } from "lucide-react";

export const LandingPage = ({ onGoToAuth, onGoToCandidate, onTestCodeSubmit }) => {
  const [testCodeInput, setTestCodeInput] = useState("");

  const handleCodeSubmit = (e) => {
    e.preventDefault();
    if (testCodeInput.trim()) {
      let clean = testCodeInput.trim().split("?")[0].split("#")[0];
      if (clean.includes("/")) {
        const parts = clean.split("/").filter(Boolean);
        clean = parts[parts.length - 1];
      }
      onTestCodeSubmit(clean.trim());
    }
  };

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "60px 24px 100px" }}>
      {/* Hero Section */}
      <div style={{ textAlign: "center", marginBottom: "70px", position: "relative" }}>
        {/* Soft Ambient Purple Glow Behind Hero */}
        <div style={{
          position: "absolute",
          top: "10%",
          left: "50%",
          transform: "translateX(-50%)",
          width: "480px",
          height: "220px",
          background: "radial-gradient(circle, rgba(124, 58, 237, 0.22) 0%, rgba(192, 38, 211, 0.08) 50%, transparent 70%)",
          filter: "blur(50px)",
          pointerEvents: "none",
          zIndex: 0,
        }} />

        <div style={{
          position: "relative",
          zIndex: 1,
        }}>
          <div style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            padding: "6px 16px",
            borderRadius: "var(--radius-full)",
            background: "rgba(124, 58, 237, 0.12)",
            border: "1px solid rgba(124, 58, 237, 0.3)",
            color: "#c084fc",
            fontSize: "0.85rem",
            fontWeight: 600,
            marginBottom: "24px",
            boxShadow: "0 0 20px rgba(124, 58, 237, 0.18)",
          }}>
            <Sparkles size={16} />
            Context-Aware Multimodal AI Proctoring
          </div>

          <h1 style={{
            fontSize: "clamp(2.5rem, 5vw, 4.2rem)",
            fontWeight: 800,
            lineHeight: 1.15,
            letterSpacing: "-0.03em",
            marginBottom: "20px",
            color: "#ffffff",
          }}>
            Next-Gen AI Testing & <br />
            <span className="text-gradient">Intelligent Proctoring</span>
          </h1>

          <p style={{
            maxWidth: "680px",
            margin: "0 auto 36px",
            fontSize: "1.15rem",
            color: "var(--text-secondary)",
            lineHeight: 1.6,
          }}>
            IntelliScreen evaluates technical assessments with server-side AI computer vision, voice activity detection, and an explainable 3-strike audit system.
          </p>

          {/* Action Buttons & Candidate Quick Link Form */}
          <div style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "center",
            gap: "16px",
            marginBottom: "36px",
          }}>
            <button
              onClick={onGoToAuth}
              className="btn-primary"
              style={{ padding: "14px 28px", fontSize: "1rem" }}
            >
              Educator / Admin Portal
              <ArrowRight size={18} />
            </button>
            
            <button
              onClick={onGoToCandidate}
              className="btn-secondary"
              style={{ padding: "14px 28px", fontSize: "1rem" }}
            >
              Candidate Portal
            </button>
          </div>

          {/* Enter Invitation Code Direct Input */}
          <form onSubmit={handleCodeSubmit} style={{
            maxWidth: "460px",
            margin: "0 auto",
            display: "flex",
            gap: "8px",
            background: "#181818",
            padding: "6px",
            borderRadius: "var(--radius-lg)",
            border: "1px solid var(--border-strong)",
            boxShadow: "0 4px 20px rgba(0, 0, 0, 0.4)",
          }}>
            <input
              type="text"
              placeholder="Have a test link code? (e.g. 191b853741e5)"
              value={testCodeInput}
              onChange={(e) => setTestCodeInput(e.target.value)}
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                padding: "10px 14px",
                color: "#fff",
                fontSize: "0.9rem",
                outline: "none",
              }}
            />
            <button
              type="submit"
              className="btn-primary"
              style={{ padding: "10px 20px", fontSize: "0.9rem" }}
            >
              Open Exam
            </button>
          </form>
        </div>
      </div>

      {/* Feature Architecture Grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
        gap: "24px",
        marginTop: "40px",
      }}>
        <div className="glass-panel" style={{ padding: "32px", background: "linear-gradient(180deg, #1d1d1d 0%, #171717 100%)" }}>
          <div style={{
            width: "48px",
            height: "48px",
            borderRadius: "12px",
            background: "rgba(124, 58, 237, 0.15)",
            border: "1px solid rgba(124, 58, 237, 0.35)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "20px",
          }}>
            <Eye size={24} color="#c084fc" />
          </div>
          <h3 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "10px", color: "#fff" }}>
            Vision Anomaly Perception
          </h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.6 }}>
            Server-side AI runs continuous face verification, gaze angle tracking, head pose deviation, and unauthorized multi-person detection.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: "32px", background: "linear-gradient(180deg, #1d1d1d 0%, #171717 100%)" }}>
          <div style={{
            width: "48px",
            height: "48px",
            borderRadius: "12px",
            background: "rgba(192, 38, 211, 0.15)",
            border: "1px solid rgba(192, 38, 211, 0.35)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "20px",
          }}>
            <Mic size={24} color="#e879f9" />
          </div>
          <h3 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "10px", color: "#fff" }}>
            Voice Activity & Audio AI
          </h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.6 }}>
            Microphone buffer slices are analyzed for secondary speakers, background whispers, and unauthorized verbal communication.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: "32px", background: "linear-gradient(180deg, #1d1d1d 0%, #171717 100%)" }}>
          <div style={{
            width: "48px",
            height: "48px",
            borderRadius: "12px",
            background: "rgba(245, 158, 11, 0.15)",
            border: "1px solid rgba(245, 158, 11, 0.35)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: "20px",
          }}>
            <AlertTriangle size={24} color="var(--warning)" />
          </div>
          <h3 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "10px", color: "#fff" }}>
            Server-Side 3-Strike Rule
          </h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.6 }}>
            Strikes 1 & 2 deliver interactive warning dialogs to guide candidate compliance. Strike 3 locks the session and terminates the exam immediately.
          </p>
        </div>
      </div>
    </div>
  );
};
