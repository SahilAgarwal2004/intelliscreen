import React, { useState, useEffect } from "react";
import { AlertTriangle, ShieldAlert, CheckCircle } from "lucide-react";

export const StrikeWarningModal = ({ warning, onAcknowledge }) => {
  const [countdown, setCountdown] = useState(5);

  useEffect(() => {
    if (countdown > 0) {
      const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [countdown]);

  if (!warning) return null;

  return (
    <div style={{
      position: "fixed",
      inset: 0,
      zIndex: 999,
      background: "rgba(0, 0, 0, 0.85)",
      backdropFilter: "blur(12px)",
      WebkitBackdropFilter: "blur(12px)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "20px",
    }}>
      <div style={{
        maxWidth: "520px",
        width: "100%",
        background: "linear-gradient(180deg, #1f140e 0%, #150f0c 100%)",
        border: "2px solid var(--warning)",
        borderRadius: "var(--radius-xl)",
        padding: "32px",
        boxShadow: "0 0 50px rgba(245, 158, 11, 0.4)",
        textAlign: "center",
        animation: "pulseGlow 2s infinite ease-in-out",
      }}>
        {/* Warning Icon Badge */}
        <div style={{
          width: "64px",
          height: "64px",
          borderRadius: "50%",
          background: "rgba(245, 158, 11, 0.2)",
          border: "2px solid var(--warning)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          margin: "0 auto 20px",
          boxShadow: "0 0 25px rgba(245, 158, 11, 0.5)",
        }}>
          <ShieldAlert size={36} color="var(--warning)" />
        </div>

        <span className="badge badge-warning" style={{ fontSize: "0.85rem", padding: "6px 14px" }}>
          PROCTORING WARNING — STRIKE {warning.strike} OF 3
        </span>

        <h2 style={{ fontSize: "1.6rem", fontWeight: 800, marginTop: "16px", color: "#fef3c7" }}>
          Suspicious Activity Flagged
        </h2>

        <div style={{
          background: "rgba(0, 0, 0, 0.4)",
          border: "1px solid rgba(245, 158, 11, 0.3)",
          borderRadius: "var(--radius-md)",
          padding: "16px",
          margin: "20px 0",
          textAlign: "left",
        }}>
          <div style={{ fontSize: "0.85rem", color: "#fde68a", fontWeight: 600, marginBottom: "4px" }}>
            VIOLATION DETAILS:
          </div>
          <p style={{ color: "#fff", fontSize: "0.95rem" }}>
            {warning.message || "An irregularity in webcam view or gaze was detected."}
          </p>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "8px" }}>
            Violation Code: <code style={{ color: "#c084fc", fontFamily: "var(--font-mono)" }}>{warning.anomalyType}</code>
          </div>
        </div>

        <p style={{ fontSize: "0.875rem", color: "#d1d5db", marginBottom: "24px", lineHeight: "1.5" }}>
          Please keep your eyes focused on the screen and ensure only you are present in the frame.
          <strong style={{ display: "block", color: "var(--danger)", marginTop: "6px" }}>
            Receiving 3 strikes will permanently lock and terminate your test attempt.
          </strong>
        </p>

        <button
          onClick={onAcknowledge}
          disabled={countdown > 0}
          className="btn-primary"
          style={{
            width: "100%",
            padding: "14px",
            fontSize: "1rem",
            background: countdown > 0 ? "rgba(255, 255, 255, 0.1)" : "linear-gradient(135deg, var(--warning), #d97706)",
            color: countdown > 0 ? "var(--text-muted)" : "#000",
            fontWeight: 700,
            cursor: countdown > 0 ? "not-allowed" : "pointer",
            boxShadow: countdown > 0 ? "none" : "0 0 25px rgba(245, 158, 11, 0.5)",
          }}
        >
          {countdown > 0 ? `Please read warning (${countdown}s)...` : "I Understand & Resume Assessment"}
        </button>
      </div>
    </div>
  );
};
