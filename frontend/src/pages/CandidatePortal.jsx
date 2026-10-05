import React, { useState, useEffect, useRef } from "react";
import {
  Shield,
  Camera,
  Mic,
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Clock,
  RefreshCw,
  UserCheck,
  User,
} from "lucide-react";
import { candidateApi } from "../api/client";

// ─── Step constants ──────────────────────────────────────────────────────────
const STEP_CODE    = 1;   // Enter invitation code
const STEP_INFO    = 2;   // Candidate details
const STEP_CAMERA  = 3;   // Grant camera + capture photo
const STEP_CONFIRM = 4;   // "Is this you?" confirmation
const STEP_READY   = 5;   // About to start (submit happens here)

export const CandidatePortal = ({ initialShareableLink, onStartExam }) => {
  const [step, setStep] = useState(STEP_CODE);

  // Step 1 state
  const [shareableLink, setShareableLink] = useState(initialShareableLink || "");
  const [testInfo, setTestInfo]           = useState(null);

  // Step 2 state
  const [candidateName,  setCandidateName]  = useState("");
  const [candidateEmail, setCandidateEmail] = useState("");
  const [candidatePhone, setCandidatePhone] = useState("");

  // Step 3 state
  const [cameraGranted,  setCameraGranted]  = useState(false);
  const [micGranted,     setMicGranted]     = useState(false);
  const [captureState,   setCaptureState]   = useState("idle"); // idle | capturing | captured | failed
  const [baselinePhoto,  setBaselinePhoto]  = useState(null);

  // Step 4 state
  const [photoConfirmed, setPhotoConfirmed] = useState(false);

  const [error,   setError]   = useState("");
  const [loading, setLoading] = useState(false);

  const previewVideoRef = useRef(null);
  const mediaStreamRef  = useRef(null);

  // ── Helpers ────────────────────────────────────────────────────────────────

  const extractCode = (raw) => {
    if (!raw) return "";
    let clean = raw.trim().split("?")[0].split("#")[0];
    if (clean.includes("/")) {
      const parts = clean.split("/").filter(Boolean);
      clean = parts[parts.length - 1];
    }
    return clean.trim();
  };

  // ── Step 1: Load test ─────────────────────────────────────────────────────

  const fetchTestDetails = async (code) => {
    const cleanCode = extractCode(code);
    if (!cleanCode) return;
    setError("");
    setLoading(true);
    try {
      const res = await candidateApi.getTestInfo(cleanCode);
      if (res.success) {
        setTestInfo(res.data);
        setStep(STEP_INFO);
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

  // ── Step 2 → 3 ───────────────────────────────────────────────────────────

  const handleInfoNext = (e) => {
    e.preventDefault();
    if (!candidateName.trim() || !candidateEmail.trim()) {
      setError("Full name and email are required.");
      return;
    }
    setError("");
    setStep(STEP_CAMERA);
  };

  // ── Step 3: Camera permission + photo capture ─────────────────────────────

  const requestMediaPermissions = async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: true,
      });
      mediaStreamRef.current = stream;
      if (previewVideoRef.current) {
        previewVideoRef.current.srcObject = stream;
        previewVideoRef.current.play().catch(() => {});
      }
      setCameraGranted(true);
      setMicGranted(true);
      // Auto-capture after exposure settles
      setTimeout(() => capturePhoto(), 900);
    } catch (err) {
      console.error("Media permission error:", err);
      setError(
        "Please grant camera and microphone permissions to proceed with this AI-proctored assessment."
      );
    }
  };

  const capturePhoto = () => {
    if (!previewVideoRef.current) return;
    const video = previewVideoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      setCaptureState("failed");
      return;
    }
    setCaptureState("capturing");
    const canvas = document.createElement("canvas");
    canvas.width  = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d");
    // Mirror the capture to match what the candidate sees (natural selfie orientation)
    ctx.translate(640, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, 640, 480);
    const photo = canvas.toDataURL("image/jpeg", 0.88);
    setBaselinePhoto(photo);
    setPhotoConfirmed(false);
    setCaptureState("captured");
  };

  const retakePhoto = () => {
    setBaselinePhoto(null);
    setPhotoConfirmed(false);
    setCaptureState("idle");
    setTimeout(() => capturePhoto(), 400);
  };

  const proceedToConfirm = () => {
    setStep(STEP_CONFIRM);
  };

  // ── Step 4: Confirm identity ──────────────────────────────────────────────

  const handleConfirmIdentity = () => {
    setPhotoConfirmed(true);
    setStep(STEP_READY);
  };

  const handleRetakeFromConfirm = () => {
    setBaselinePhoto(null);
    setPhotoConfirmed(false);
    setCaptureState("idle");
    setStep(STEP_CAMERA);
    setTimeout(() => capturePhoto(), 800);
  };

  // ── Step 5: Start exam ────────────────────────────────────────────────────

  const handleStartExam = async (e) => {
    e.preventDefault();
    const cleanCode = extractCode(shareableLink);
    if (!testInfo || !cleanCode) {
      setError("Please enter a valid test link code.");
      return;
    }
    if (!cameraGranted || !micGranted) {
      setError("Webcam and microphone access are strictly required for proctoring.");
      return;
    }
    if (!photoConfirmed) {
      setError("Please confirm your identity photo before starting.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      if (document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen().catch(() => {});
      }
      const res = await candidateApi.startAttempt({
        shareableLink: cleanCode,
        name:          candidateName.trim(),
        email:         candidateEmail.trim(),
        phone:         candidatePhone.trim() || undefined,
        baselinePhoto: baselinePhoto || undefined,
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

  // ── Shared header ─────────────────────────────────────────────────────────

  const StepIndicator = () => (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", marginBottom: "28px" }}>
      {[
        { n: 1, label: "Code"    },
        { n: 2, label: "Info"    },
        { n: 3, label: "Camera"  },
        { n: 4, label: "Confirm" },
        { n: 5, label: "Start"   },
      ].map(({ n, label }, i, arr) => (
        <React.Fragment key={n}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "4px" }}>
            <div style={{
              width: "32px", height: "32px", borderRadius: "50%",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: "0.8rem", fontWeight: 700,
              background: step === n
                ? "linear-gradient(135deg, #7c3aed, #c026d3)"
                : step > n ? "rgba(52,211,153,0.25)" : "rgba(255,255,255,0.08)",
              border: step === n ? "none" : step > n ? "1px solid #34d399" : "1px solid rgba(255,255,255,0.15)",
              color: step === n ? "#fff" : step > n ? "#34d399" : "var(--text-muted)",
            }}>
              {step > n ? "✓" : n}
            </div>
            <span style={{ fontSize: "0.68rem", color: step >= n ? "var(--text-secondary)" : "var(--text-muted)" }}>
              {label}
            </span>
          </div>
          {i < arr.length - 1 && (
            <div style={{
              flex: 1, height: "1px", maxWidth: "40px",
              background: step > n ? "rgba(52,211,153,0.4)" : "rgba(255,255,255,0.1)",
              marginBottom: "18px",
            }} />
          )}
        </React.Fragment>
      ))}
    </div>
  );

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div style={{ maxWidth: "760px", margin: "40px auto 100px", padding: "0 24px" }}>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "28px" }}>
        <div style={{
          width: "48px", height: "48px", borderRadius: "12px",
          background: "linear-gradient(135deg, #7c3aed 0%, #c026d3 100%)",
          display: "inline-flex", alignItems: "center", justifyContent: "center",
          boxShadow: "0 0 24px rgba(124,58,237,0.45)", marginBottom: "14px",
        }}>
          <Shield size={24} color="#fff" />
        </div>
        <h1 style={{ fontSize: "1.85rem", fontWeight: 800, color: "#fff" }}>
          Candidate Assessment Check-In
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginTop: "6px" }}>
          Complete each step to begin your AI-proctored examination.
        </p>
      </div>

      <StepIndicator />

      {/* Global error */}
      {error && (
        <div style={{
          background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.3)",
          borderRadius: "var(--radius-md)", padding: "14px 18px",
          color: "#fca5a5", fontSize: "0.9rem",
          display: "flex", alignItems: "center", gap: "10px", marginBottom: "20px",
        }}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* ── STEP 1: Enter code ───────────────────────────────────────────── */}
      {step === STEP_CODE && (
        <div className="glass-panel" style={{ padding: "32px", background: "#1a1a1a" }}>
          <h3 style={{ fontSize: "1.15rem", fontWeight: 700, marginBottom: "8px", color: "#fff" }}>
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
              onKeyDown={(e) => e.key === "Enter" && fetchTestDetails(shareableLink)}
              className="form-input"
              style={{ flex: 1 }}
            />
            <button
              onClick={() => fetchTestDetails(shareableLink)}
              disabled={loading || !shareableLink}
              className="btn-primary"
            >
              {loading ? "Searching…" : "Lookup"}
            </button>
          </div>
        </div>
      )}

      {/* ── STEP 2: Candidate info ───────────────────────────────────────── */}
      {step === STEP_INFO && testInfo && (
        <form onSubmit={handleInfoNext} style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          {/* Test card */}
          <div className="glass-panel" style={{ padding: "22px", border: "1px solid rgba(124,58,237,0.35)", background: "#1a1a1a" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
              <div>
                <span className="badge badge-indigo" style={{ marginBottom: "6px" }}>VERIFIED EXAM</span>
                <h2 style={{ fontSize: "1.3rem", fontWeight: 800, color: "#fff" }}>{testInfo.title}</h2>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.88rem", marginTop: "4px" }}>
                  {testInfo.description || "Answer all multiple-choice questions within the allocated time."}
                </p>
              </div>
              <button type="button" onClick={() => { setStep(STEP_CODE); setTestInfo(null); }}
                style={{ fontSize: "0.8rem", color: "var(--text-muted)", textDecoration: "underline" }}>
                Change
              </button>
            </div>
            <div style={{ display: "flex", gap: "18px", fontSize: "0.82rem", color: "var(--text-muted)", borderTop: "1px solid var(--border-subtle)", paddingTop: "12px" }}>
              <span style={{ display: "flex", alignItems: "center", gap: "5px", color: "#e879f9" }}>
                <Clock size={14} /><strong>{testInfo.durationMinutes} min</strong>
              </span>
              <span>•</span>
              <span><strong>{testInfo.questionsCount}</strong> Questions</span>
              <span>•</span>
              <span><strong>{testInfo.totalMarks}</strong> Marks</span>
            </div>
          </div>

          {/* Fields */}
          <div className="glass-panel" style={{ padding: "26px", background: "#1a1a1a" }}>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "16px", color: "#fff" }}>
              Candidate Information
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
              <div style={{ gridColumn: "span 2" }}>
                <label style={{ display: "block", fontSize: "0.83rem", fontWeight: 600, marginBottom: "6px", color: "var(--text-secondary)" }}>
                  Full Legal Name *
                </label>
                <input type="text" required placeholder="Jane Doe"
                  value={candidateName} onChange={(e) => setCandidateName(e.target.value)}
                  className="form-input" />
              </div>
              <div>
                <label style={{ display: "block", fontSize: "0.83rem", fontWeight: 600, marginBottom: "6px", color: "var(--text-secondary)" }}>
                  Email Address *
                </label>
                <input type="email" required placeholder="jane@university.edu"
                  value={candidateEmail} onChange={(e) => setCandidateEmail(e.target.value)}
                  className="form-input" />
              </div>
              <div>
                <label style={{ display: "block", fontSize: "0.83rem", fontWeight: 600, marginBottom: "6px", color: "var(--text-secondary)" }}>
                  Phone (Optional)
                </label>
                <input type="tel" placeholder="+1 555 000 0000"
                  value={candidatePhone} onChange={(e) => setCandidatePhone(e.target.value)}
                  className="form-input" />
              </div>
            </div>
          </div>

          <button type="submit" className="btn-primary" style={{ padding: "14px", fontSize: "1rem" }}>
            Continue to Camera Setup <ArrowRight size={18} />
          </button>
        </form>
      )}

      {/* ── STEP 3: Camera + photo capture ──────────────────────────────── */}
      {step === STEP_CAMERA && (
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <div className="glass-panel" style={{ padding: "26px", background: "#1a1a1a" }}>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "6px", color: "#fff" }}>
              Identity Photo & Hardware Check
            </h3>
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", marginBottom: "20px" }}>
              IntelliScreen will take a reference photo to verify your identity throughout the exam.
              Sit in a well-lit area and look directly at the camera.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: "20px", alignItems: "start" }}>
              {/* Live camera feed */}
              <div style={{
                position: "relative", width: "100%", aspectRatio: "4/3",
                background: "#141414", borderRadius: "var(--radius-md)", overflow: "hidden",
                border: captureState === "captured" ? "2px solid #34d399" : "1px solid var(--border-strong)",
              }}>
                <video ref={previewVideoRef} muted playsInline
                  style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }} />

                {/* Overlay face guide */}
                {cameraGranted && captureState !== "captured" && (
                  <div style={{
                    position: "absolute", inset: 0, display: "flex",
                    alignItems: "center", justifyContent: "center", pointerEvents: "none",
                  }}>
                    <div style={{
                      width: "42%", aspectRatio: "3/4",
                      border: "2px dashed rgba(124,58,237,0.55)",
                      borderRadius: "50%",
                    }} />
                  </div>
                )}

                {/* Capturing flash overlay */}
                {captureState === "capturing" && (
                  <div style={{
                    position: "absolute", inset: 0,
                    background: "rgba(255,255,255,0.25)",
                    animation: "none",
                  }} />
                )}

                {!cameraGranted && (
                  <div style={{
                    position: "absolute", inset: 0, display: "flex",
                    flexDirection: "column", alignItems: "center", justifyContent: "center",
                    background: "rgba(18,18,18,0.95)", padding: "20px", textAlign: "center",
                  }}>
                    <Camera size={36} color="var(--text-muted)" style={{ marginBottom: "10px" }} />
                    <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                      Camera not started
                    </span>
                  </div>
                )}

                {/* Captured badge */}
                {captureState === "captured" && (
                  <div style={{
                    position: "absolute", bottom: "8px", left: "50%", transform: "translateX(-50%)",
                    background: "rgba(16,185,129,0.9)", color: "#fff",
                    fontSize: "0.72rem", fontWeight: 700, padding: "4px 12px",
                    borderRadius: "99px", letterSpacing: "0.04em",
                  }}>
                    ✓ PHOTO CAPTURED
                  </div>
                )}
              </div>

              {/* Checklist + buttons */}
              <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "0.88rem", color: "#e5e5e5" }}>
                  <CheckCircle2 size={17} color={cameraGranted ? "var(--success)" : "var(--text-muted)"} />
                  <span>Webcam Stream</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "0.88rem", color: "#e5e5e5" }}>
                  <CheckCircle2 size={17} color={micGranted ? "var(--success)" : "var(--text-muted)"} />
                  <span>Microphone</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "0.88rem", color: "#e5e5e5" }}>
                  <CheckCircle2 size={17} color={captureState === "captured" ? "var(--success)" : "var(--text-muted)"} />
                  <span>Reference Photo</span>
                </div>

                {!cameraGranted ? (
                  <button type="button" onClick={requestMediaPermissions}
                    className="btn-secondary"
                    style={{ marginTop: "6px", width: "100%", padding: "10px" }}>
                    <Camera size={15} style={{ marginRight: "6px" }} />
                    Grant Camera & Mic
                  </button>
                ) : captureState === "captured" ? (
                  <button type="button" onClick={retakePhoto}
                    className="btn-secondary"
                    style={{ width: "100%", padding: "8px", fontSize: "0.82rem" }}>
                    <RefreshCw size={14} style={{ marginRight: "5px" }} />
                    Retake Photo
                  </button>
                ) : (
                  <button type="button" onClick={capturePhoto} disabled={captureState === "capturing"}
                    className="btn-secondary"
                    style={{ width: "100%", padding: "8px", fontSize: "0.82rem" }}>
                    {captureState === "capturing" ? "Capturing…" : "Capture Photo"}
                  </button>
                )}

                <p style={{ fontSize: "0.76rem", color: "var(--text-muted)", lineHeight: 1.5 }}>
                  Make sure your full face is visible, well-lit, and centred in the oval guide.
                </p>
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={proceedToConfirm}
            disabled={captureState !== "captured"}
            className="btn-primary"
            style={{ padding: "14px", fontSize: "1rem", opacity: captureState !== "captured" ? 0.45 : 1 }}>
            Review Identity Photo <ArrowRight size={18} />
          </button>
        </div>
      )}

      {/* ── STEP 4: "Is this you?" confirmation ─────────────────────────── */}
      {step === STEP_CONFIRM && baselinePhoto && (
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <div className="glass-panel" style={{ padding: "28px", background: "#1a1a1a" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "18px" }}>
              <UserCheck size={22} color="#c026d3" />
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#fff" }}>
                Is this you?
              </h3>
            </div>
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", marginBottom: "22px" }}>
              This photo will be used as your identity baseline throughout the exam.
              The AI proctoring system will compare live frames against this reference.
              If this does not clearly show your face, please retake it.
            </p>

            {/* Photo preview */}
            <div style={{
              display: "flex", justifyContent: "center", marginBottom: "24px",
            }}>
              <div style={{
                position: "relative",
                width: "220px", height: "165px",
                borderRadius: "var(--radius-md)", overflow: "hidden",
                border: "2px solid rgba(124,58,237,0.5)",
                boxShadow: "0 0 30px rgba(124,58,237,0.25)",
              }}>
                <img
                  src={baselinePhoto}
                  alt="Reference photo"
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                />
                <div style={{
                  position: "absolute", bottom: "6px", left: "50%", transform: "translateX(-50%)",
                  background: "rgba(0,0,0,0.7)", color: "#e879f9",
                  fontSize: "0.68rem", fontWeight: 700, padding: "3px 10px",
                  borderRadius: "99px", letterSpacing: "0.06em", whiteSpace: "nowrap",
                }}>
                  REFERENCE PHOTO
                </div>
              </div>
            </div>

            {/* Tips */}
            <div style={{
              background: "rgba(124,58,237,0.08)", border: "1px solid rgba(124,58,237,0.2)",
              borderRadius: "var(--radius-sm)", padding: "12px 16px",
              fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "22px",
              lineHeight: 1.6,
            }}>
              ✓ &nbsp;Face clearly visible &nbsp;|&nbsp; ✓ &nbsp;Well lit &nbsp;|&nbsp; ✓ &nbsp;No obstructions (hat/sunglasses)
            </div>

            <div style={{ display: "flex", gap: "12px" }}>
              <button type="button" onClick={handleRetakeFromConfirm}
                className="btn-secondary"
                style={{ flex: 1, padding: "12px", fontSize: "0.95rem" }}>
                <RefreshCw size={16} style={{ marginRight: "6px" }} />
                Retake Photo
              </button>
              <button type="button" onClick={handleConfirmIdentity}
                className="btn-primary"
                style={{ flex: 1.5, padding: "12px", fontSize: "0.95rem" }}>
                <UserCheck size={16} style={{ marginRight: "6px" }} />
                Yes, This Is Me — Confirm
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── STEP 5: Ready to start ───────────────────────────────────────── */}
      {step === STEP_READY && (
        <form onSubmit={handleStartExam} style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <div className="glass-panel" style={{ padding: "28px", background: "#1a1a1a" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
              <div style={{
                width: "40px", height: "40px", borderRadius: "50%",
                background: "rgba(52,211,153,0.15)", display: "flex",
                alignItems: "center", justifyContent: "center",
              }}>
                <CheckCircle2 size={22} color="#34d399" />
              </div>
              <div>
                <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "#fff" }}>
                  All checks passed — you're ready
                </h3>
                <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                  Your identity has been captured and verified. The exam will begin in fullscreen mode.
                </p>
              </div>
            </div>

            {/* Summary */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginTop: "10px" }}>
              {[
                { icon: <User size={15} />, label: "Candidate",      value: candidateName },
                { icon: <CheckCircle2 size={15} color="#34d399" />, label: "Camera & Mic", value: "Active" },
                { icon: <UserCheck size={15} color="#e879f9" />,     label: "Identity",    value: "Confirmed ✓" },
                { icon: <Clock size={15} />,                         label: "Duration",    value: `${testInfo?.durationMinutes} min` },
              ].map(({ icon, label, value }) => (
                <div key={label} style={{
                  background: "rgba(255,255,255,0.04)", borderRadius: "var(--radius-sm)",
                  padding: "10px 14px", display: "flex", alignItems: "center", gap: "10px",
                }}>
                  <span style={{ color: "var(--text-muted)" }}>{icon}</span>
                  <div>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: 600 }}>{label}</div>
                    <div style={{ fontSize: "0.85rem", color: "#e5e5e5", fontWeight: 600 }}>{value}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {error && (
            <div style={{
              background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.3)",
              borderRadius: "var(--radius-md)", padding: "12px 16px", color: "#fca5a5",
              fontSize: "0.88rem", display: "flex", alignItems: "center", gap: "8px",
            }}>
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          <button type="submit" disabled={loading} className="btn-primary"
            style={{
              padding: "16px", fontSize: "1.05rem",
              boxShadow: "0 0 30px rgba(124,58,237,0.45)",
            }}>
            {loading ? "Initializing Proctored Session…" : "Begin Proctored Examination"}
            {!loading && <ArrowRight size={20} />}
          </button>

          <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", textAlign: "center" }}>
            By clicking above you confirm that you are the registered candidate and agree to AI-based proctoring monitoring for the duration of this assessment.
          </p>
        </form>
      )}
    </div>
  );
};
