import React, { useState, useEffect, useRef } from "react";
import { Shield, Camera, Mic, AlertCircle, ArrowRight, CheckCircle2, Clock } from "lucide-react";
import { candidateApi } from "../api/client";

export const CandidatePortal = ({ initialShareableLink, onStartExam }) => {
  const [shareableLink, setShareableLink] = useState(initialShareableLink || "");
  const [testInfo, setTestInfo] = useState(null);
  const [candidateName, setCandidateName] = useState("");
  const [candidateEmail, setCandidateEmail] = useState("");
  const [candidatePhone, setCandidatePhone] = useState("");
  const [cameraGranted, setCameraGranted] = useState(false);
  const [micGranted, setMicGranted] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const previewVideoRef = useRef(null);

  // Load test overview when link is entered
  const fetchTestDetails = async (code) => {
    if (!code) return;
    setError("");
    setLoading(true);
    try {
      const res = await candidateApi.getTestInfo(code.trim());
      if (res.success) {
        setTestInfo(res.data);
      }
    } catch (err) {
      setTestInfo(null);
      setError(err.message || "Invalid or inactive test link");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (initialShareableLink) {
      fetchTestDetails(initialShareableLink);
    }
  }, [initialShareableLink]);

  // Request Camera & Microphone Permissions
  const requestMediaPermissions = async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      if (previewVideoRef.current) {
        previewVideoRef.current.srcObject = stream;
        previewVideoRef.current.play().catch(() => {});
      }
      setCameraGranted(true);
      setMicGranted(true);
    } catch (err) {
      console.error("Media permission error:", err);
      setError("Please grant camera and microphone permissions to proceed with this AI-proctored assessment.");
    }
  };

  const handleStartExam = async (e) => {
    e.preventDefault();
    if (!testInfo) {
      setError("Please enter a valid test link code.");
      return;
    }
    if (!cameraGranted || !micGranted) {
      setError("Webcam and microphone access are strictly required for proctoring.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      // Request fullscreen mode
      if (document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen().catch(() => {});
      }

      const res = await candidateApi.startAttempt({
        shareableLink: shareableLink.trim(),
        name: candidateName.trim(),
        email: candidateEmail.trim(),
        phone: candidatePhone.trim() || undefined,
      });

      if (res.success) {
        onStartExam(res.data);
      }
    } catch (err) {
      setError(err.message || "Failed to start assessment attempt");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: "800px", margin: "40px auto 100px", padding: "0 24px" }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "36px" }}>
        <div style={{
          width: "48px",
          height: "48px",
          borderRadius: "12px",
          background: "linear-gradient(135deg, #7c3aed 0%, #c026d3 100%)",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: "0 0 24px rgba(124, 58, 237, 0.45)",
          marginBottom: "16px",
        }}>
          <Shield size={24} color="#fff" />
        </div>
        <h1 style={{ fontSize: "2rem", fontWeight: 800, color: "#fff" }}>Candidate Assessment Check-In</h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", marginTop: "6px" }}>
          Verify your hardware, grant proctoring permissions, and begin your test.
        </p>
      </div>

      {error && (
        <div style={{
          background: "rgba(239, 68, 68, 0.12)",
          border: "1px solid rgba(239, 68, 68, 0.3)",
          borderRadius: "var(--radius-md)",
          padding: "14px 18px",
          color: "#fca5a5",
          fontSize: "0.9rem",
          display: "flex",
          alignItems: "center",
          gap: "10px",
          marginBottom: "24px",
        }}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* Step 1: Test Code Lookup */}
      {!testInfo && (
        <div className="glass-panel" style={{ padding: "32px", marginBottom: "28px", background: "#1a1a1a" }}>
          <h3 style={{ fontSize: "1.2rem", fontWeight: 700, marginBottom: "8px", color: "#fff" }}>
            Enter Assessment Invitation Code
          </h3>
          <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", marginBottom: "20px" }}>
            Paste the invitation code or link shared by your institution or recruiter.
          </p>

          <div style={{ display: "flex", gap: "10px" }}>
            <input
              type="text"
              placeholder="e.g. a19dbcf7cbec"
              value={shareableLink}
              onChange={(e) => setShareableLink(e.target.value)}
              className="form-input"
              style={{ flex: 1 }}
            />
            <button
              onClick={() => fetchTestDetails(shareableLink)}
              disabled={loading || !shareableLink}
              className="btn-primary"
            >
              {loading ? "Searching..." : "Lookup Assessment"}
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Test Overview & Candidate Form */}
      {testInfo && (
        <form onSubmit={handleStartExam} style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          {/* Test Card */}
          <div className="glass-panel" style={{ padding: "24px", border: "1px solid rgba(124, 58, 237, 0.35)", background: "#1a1a1a" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
              <div>
                <span className="badge badge-indigo" style={{ marginBottom: "6px" }}>VERIFIED EXAM</span>
                <h2 style={{ fontSize: "1.4rem", fontWeight: 800, color: "#fff" }}>{testInfo.title}</h2>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginTop: "4px" }}>
                  {testInfo.description || "Answer all multiple-choice questions within the allocated time limit."}
                </p>
              </div>

              <button
                type="button"
                onClick={() => setTestInfo(null)}
                style={{ fontSize: "0.8rem", color: "var(--text-muted)", textDecoration: "underline" }}
              >
                Change Code
              </button>
            </div>

            <div style={{ display: "flex", gap: "20px", fontSize: "0.85rem", color: "var(--text-muted)", borderTop: "1px solid var(--border-subtle)", paddingTop: "14px" }}>
              <span style={{ display: "flex", alignItems: "center", gap: "6px", color: "#e879f9" }}>
                <Clock size={15} />
                <strong>{testInfo.durationMinutes} Minutes</strong>
              </span>
              <span>•</span>
              <span><strong>{testInfo.questionsCount}</strong> Questions</span>
              <span>•</span>
              <span><strong>{testInfo.totalMarks}</strong> Total Marks</span>
              <span>•</span>
              <span>Max Attempts: <strong>{testInfo.maxAttempts}</strong></span>
            </div>
          </div>

          {/* Candidate Registration Inputs */}
          <div className="glass-panel" style={{ padding: "28px", background: "#1a1a1a" }}>
            <h3 style={{ fontSize: "1.15rem", fontWeight: 700, marginBottom: "16px", color: "#fff" }}>
              Candidate Information
            </h3>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
              <div style={{ gridColumn: "span 2" }}>
                <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "6px", color: "var(--text-secondary)" }}>
                  Full Legal Name *
                </label>
                <input
                  type="text"
                  required
                  placeholder="Jane Doe"
                  value={candidateName}
                  onChange={(e) => setCandidateName(e.target.value)}
                  className="form-input"
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "6px", color: "var(--text-secondary)" }}>
                  Institutional Email Address *
                </label>
                <input
                  type="email"
                  required
                  placeholder="jane@university.edu"
                  value={candidateEmail}
                  onChange={(e) => setCandidateEmail(e.target.value)}
                  className="form-input"
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "6px", color: "var(--text-secondary)" }}>
                  Contact Phone (Optional)
                </label>
                <input
                  type="tel"
                  placeholder="+1 (555) 000-0000"
                  value={candidatePhone}
                  onChange={(e) => setCandidatePhone(e.target.value)}
                  className="form-input"
                />
              </div>
            </div>
          </div>

          {/* Hardware & System Checks */}
          <div className="glass-panel" style={{ padding: "28px", background: "#1a1a1a" }}>
            <h3 style={{ fontSize: "1.15rem", fontWeight: 700, marginBottom: "8px", color: "#fff" }}>
              Hardware & Sensor Calibration
            </h3>
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", marginBottom: "20px" }}>
              IntelliScreen AI requires webcam and microphone telemetry to verify candidate presence.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "20px", alignItems: "center" }}>
              {/* Preview Box */}
              <div style={{
                position: "relative",
                width: "100%",
                height: "200px",
                background: "#141414",
                borderRadius: "var(--radius-md)",
                overflow: "hidden",
                border: "1px solid var(--border-strong)",
              }}>
                <video
                  ref={previewVideoRef}
                  muted
                  playsInline
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                    transform: "scaleX(-1)",
                  }}
                />
                {!cameraGranted && (
                  <div style={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "rgba(18, 18, 18, 0.95)",
                    padding: "20px",
                    textAlign: "center",
                  }}>
                    <Camera size={36} color="var(--text-muted)" style={{ marginBottom: "10px" }} />
                    <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                      Camera preview not started
                    </span>
                  </div>
                )}
              </div>

              {/* Checklist & Grant Button */}
              <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "0.9rem", color: "#e5e5e5" }}>
                  <CheckCircle2 size={18} color={cameraGranted ? "var(--success)" : "var(--text-muted)"} />
                  <span>Webcam Presence Stream</span>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "0.9rem", color: "#e5e5e5" }}>
                  <CheckCircle2 size={18} color={micGranted ? "var(--success)" : "var(--text-muted)"} />
                  <span>Microphone Voice Detection</span>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "0.9rem", color: "#e5e5e5" }}>
                  <CheckCircle2 size={18} color="var(--success)" />
                  <span>Browser Fullscreen Locked</span>
                </div>

                {!cameraGranted ? (
                  <button
                    type="button"
                    onClick={requestMediaPermissions}
                    className="btn-secondary"
                    style={{ marginTop: "10px", width: "100%", padding: "10px" }}
                  >
                    Grant Camera & Mic Access
                  </button>
                ) : (
                  <div style={{
                    padding: "8px 12px",
                    background: "rgba(52, 211, 153, 0.12)",
                    border: "1px solid var(--success)",
                    borderRadius: "var(--radius-sm)",
                    color: "#34d399",
                    fontSize: "0.8rem",
                    fontWeight: 600,
                    textAlign: "center",
                    marginTop: "8px",
                  }}>
                    ✓ Sensors Calibrated Successfully
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Submit Action */}
          <button
            type="submit"
            disabled={loading || !cameraGranted}
            className="btn-primary"
            style={{
              padding: "16px",
              fontSize: "1.05rem",
              boxShadow: cameraGranted ? "0 0 30px rgba(124, 58, 237, 0.45)" : "none",
            }}
          >
            {loading ? "Initializing Proctored Session..." : "Begin Proctored Examination"}
            <ArrowRight size={20} />
          </button>
        </form>
      )}
    </div>
  );
};
