import React from "react";
import { CheckCircle2, ArrowRight } from "lucide-react";

export const ExamSubmitted = ({ result, onHome }) => {
  const { score = 0, totalMarks = 0, percentage = 0 } = result || {};

  return (
    <div style={{
      maxWidth: "540px",
      margin: "80px auto",
      padding: "0 20px",
      position: "relative",
    }}>
      {/* Soft Ambient Glow */}
      <div style={{
        position: "absolute",
        top: "20%",
        left: "50%",
        transform: "translateX(-50%)",
        width: "380px",
        height: "220px",
        background: "radial-gradient(circle, rgba(52, 211, 153, 0.15) 0%, rgba(124, 58, 237, 0.1) 50%, transparent 70%)",
        filter: "blur(50px)",
        pointerEvents: "none",
        zIndex: 0,
      }} />

      <div className="glass-panel" style={{ padding: "40px 32px", textAlign: "center", position: "relative", zIndex: 1, background: "#1a1a1a" }}>
        {/* Success Icon */}
        <div style={{
          width: "72px",
          height: "72px",
          borderRadius: "50%",
          background: "rgba(52, 211, 153, 0.14)",
          border: "2px solid var(--success)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          margin: "0 auto 20px",
          boxShadow: "0 0 25px rgba(52, 211, 153, 0.35)",
        }}>
          <CheckCircle2 size={38} color="var(--success)" />
        </div>

        <span className="badge badge-success" style={{ fontSize: "0.8rem", padding: "4px 12px" }}>
          EXAMINATION COMPLETED
        </span>

        <h1 style={{ fontSize: "1.85rem", fontWeight: 800, marginTop: "14px", color: "#fff" }}>
          Assessment Submitted Successfully
        </h1>

        <p style={{ color: "var(--text-secondary)", fontSize: "0.925rem", marginTop: "8px", lineHeight: 1.5 }}>
          Your responses and proctoring telemetry data have been securely submitted to the educator for final evaluation.
        </p>

        {/* Scorecard Box */}
        <div style={{
          background: "#141414",
          border: "1px solid var(--border-strong)",
          borderRadius: "var(--radius-lg)",
          padding: "24px",
          margin: "28px 0",
          display: "flex",
          justifyContent: "space-around",
          alignItems: "center",
        }}>
          <div>
            <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 600 }}>SCORE OBTAINED</div>
            <div style={{ fontSize: "2.25rem", fontWeight: 800, color: "var(--success)", marginTop: "4px" }}>
              {score} <span style={{ fontSize: "1rem", color: "var(--text-muted)" }}>/ {totalMarks}</span>
            </div>
          </div>

          <div style={{ width: "1px", height: "40px", background: "var(--border-subtle)" }} />

          <div>
            <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 600 }}>PERCENTAGE</div>
            <div style={{ fontSize: "2.25rem", fontWeight: 800, color: percentage >= 60 ? "#34d399" : "#fbbf24", marginTop: "4px" }}>
              {percentage}%
            </div>
          </div>
        </div>

        <button
          onClick={onHome}
          className="btn-primary"
          style={{ width: "100%", padding: "14px", fontSize: "0.95rem" }}
        >
          Return to Home Portal
          <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
};
