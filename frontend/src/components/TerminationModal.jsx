import React from "react";
import { Lock, XOctagon, AlertOctagon } from "lucide-react";

export const TerminationModal = ({ reason, onExit }) => {
  return (
    <div style={{
      position: "fixed",
      inset: 0,
      zIndex: 1000,
      background: "radial-gradient(ellipse at center, rgba(127, 29, 29, 0.95) 0%, rgba(10, 5, 5, 0.98) 100%)",
      backdropFilter: "blur(20px)",
      WebkitBackdropFilter: "blur(20px)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "24px",
    }}>
      <div style={{
        maxWidth: "560px",
        width: "100%",
        background: "rgba(18, 10, 10, 0.95)",
        border: "2px solid var(--danger)",
        borderRadius: "var(--radius-xl)",
        padding: "40px",
        boxShadow: "0 0 60px rgba(239, 68, 68, 0.6)",
        textAlign: "center",
      }}>
        {/* Lock Icon */}
        <div style={{
          width: "80px",
          height: "80px",
          borderRadius: "50%",
          background: "rgba(239, 68, 68, 0.2)",
          border: "2px solid var(--danger)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          margin: "0 auto 24px",
          boxShadow: "0 0 35px rgba(239, 68, 68, 0.7)",
        }}>
          <Lock size={42} color="var(--danger)" />
        </div>

        <span className="badge badge-danger" style={{ fontSize: "0.85rem", padding: "6px 16px" }}>
          EXAM SESSION TERMINATED — 3/3 STRIKES
        </span>

        <h1 style={{ fontSize: "2rem", fontWeight: 800, marginTop: "16px", color: "#fee2e2" }}>
          Assessment Session Locked
        </h1>

        <div style={{
          background: "rgba(0, 0, 0, 0.5)",
          border: "1px solid rgba(239, 68, 68, 0.3)",
          borderRadius: "var(--radius-md)",
          padding: "18px",
          margin: "24px 0",
          textAlign: "left",
        }}>
          <div style={{ fontSize: "0.8rem", color: "#fca5a5", fontWeight: 700, marginBottom: "4px" }}>
            TERMINATION REASON:
          </div>
          <p style={{ color: "#fff", fontSize: "1rem", fontWeight: 500 }}>
            {reason || "Your assessment has been automatically terminated due to exceeding the maximum allowed proctoring violations (3 strikes)."}
          </p>
        </div>

        <p style={{ fontSize: "0.9rem", color: "#9ca3af", marginBottom: "32px", lineHeight: "1.6" }}>
          An audit report containing timestamps, webcam anomaly logs, and sensor activity has been sent to the examination administrator for review.
        </p>

        <button
          onClick={onExit}
          className="btn-danger"
          style={{
            width: "100%",
            padding: "14px",
            fontSize: "1rem",
            fontWeight: 700,
          }}
        >
          Exit Assessment Portal
        </button>
      </div>
    </div>
  );
};
