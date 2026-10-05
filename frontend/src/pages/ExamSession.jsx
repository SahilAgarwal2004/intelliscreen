import React, { useState, useEffect } from "react";
import io from "socket.io-client";
import { Clock, Shield, ChevronLeft, ChevronRight, CheckCircle2, Send } from "lucide-react";
import { candidateApi } from "../api/client";
import { ProctorCamera } from "../components/ProctorCamera";
import { StrikeWarningModal } from "../components/StrikeWarningModal";
import { TerminationModal } from "../components/TerminationModal";

export const ExamSession = ({ examData, onExamFinished, onExamTerminated }) => {
  const {
    attemptId,
    sessionId,
    test,
    questions = [],
    answers: initialAnswers = [],
    strikes: initialStrikes = 0,
    maxStrikes = 3,
  } = examData;

  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswers, setSelectedAnswers] = useState(() => {
    const map = {};
    initialAnswers.forEach((a) => {
      map[a.questionId] = a.selectedOption;
    });
    return map;
  });

  const [remainingSeconds, setRemainingSeconds] = useState(() => {
    if (test?.expiresAt) {
      const diff = Math.floor((new Date(test.expiresAt) - new Date()) / 1000);
      return Math.max(0, diff);
    }
    return (test?.durationMinutes || 30) * 60;
  });

  const [strikes, setStrikes] = useState(initialStrikes);
  const [activeWarning, setActiveWarning] = useState(null);
  const [isTerminated, setIsTerminated] = useState(false);
  const [terminationReason, setTerminationReason] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [socket, setSocket] = useState(null);

  // Initialize Socket.IO connection
  useEffect(() => {
    const newSocket = io(window.location.origin, {
      transports: ["websocket", "polling"],
    });

    newSocket.on("connect", () => {
      newSocket.emit("join_proctoring", { attemptId, sessionId });
    });

    setSocket(newSocket);

    return () => {
      newSocket.disconnect();
    };
  }, [attemptId, sessionId]);

  // Tab switch & Fullscreen violation detection
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden && socket && socket.connected) {
        socket.emit("client_incident", {
          attemptId,
          sessionId,
          incidentType: "tab_switch",
          details: { timestamp: Date.now() },
        });
      }
    };

    const handleFullscreenChange = () => {
      if (!document.fullscreenElement && socket && socket.connected) {
        socket.emit("client_incident", {
          attemptId,
          sessionId,
          incidentType: "fullscreen_exit",
          details: { timestamp: Date.now() },
        });
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    document.addEventListener("fullscreenchange", handleFullscreenChange);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, [socket, attemptId, sessionId]);

  // Countdown timer
  useEffect(() => {
    if (remainingSeconds <= 0) {
      handleFinalSubmit();
      return;
    }

    const interval = setInterval(() => {
      setRemainingSeconds((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          handleFinalSubmit();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [remainingSeconds]);

  // Handle Option Selection
  const handleSelectOption = async (optionIndex) => {
    const currentQuestion = questions[currentQuestionIndex];
    if (!currentQuestion) return;

    setSelectedAnswers((prev) => ({
      ...prev,
      [currentQuestion._id]: optionIndex,
    }));

    // Auto-save answer to backend
    try {
      await candidateApi.saveAnswer(attemptId, currentQuestion._id, optionIndex);
    } catch (err) {
      console.warn("Auto-save answer failed:", err);
    }
  };

  // Final Submit
  const handleFinalSubmit = async () => {
    if (isSubmitting) return;
    setIsSubmitting(true);
    try {
      const res = await candidateApi.submitTest(attemptId);
      if (res.success) {
        onExamFinished(res.data);
      }
    } catch (err) {
      console.error("Submission failed:", err);
      setIsSubmitting(false);
    }
  };

  // Warning Handlers from Socket
  const handleWarningIssued = (data) => {
    setStrikes(data.strike);
    setActiveWarning(data);
  };

  const handleForceTerminate = (data) => {
    setStrikes(data.strikes || 3);
    setIsTerminated(true);
    setTerminationReason(data.reason || "Maximum 3 proctoring violations exceeded.");
  };

  const currentQ = questions[currentQuestionIndex] || {};
  const isCurrentAnswered = selectedAnswers[currentQ._id] !== undefined;

  const minutes = Math.floor(remainingSeconds / 60);
  const seconds = remainingSeconds % 60;
  const isTimeCritical = remainingSeconds < 300; // Under 5 min

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", background: "var(--bg-primary)" }}>
      {/* Top Header Bar */}
      <header style={{
        background: "#161616",
        borderBottom: "1px solid var(--border-strong)",
        padding: "12px 24px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        zIndex: 40,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{
            width: "32px",
            height: "32px",
            borderRadius: "8px",
            background: "linear-gradient(135deg, #7c3aed 0%, #c026d3 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 0 12px rgba(124, 58, 237, 0.4)",
          }}>
            <Shield size={18} color="#fff" />
          </div>
          <div>
            <h2 style={{ fontSize: "1.05rem", fontWeight: 700, color: "#fff" }}>{test?.title}</h2>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Question {currentQuestionIndex + 1} of {questions.length}
            </div>
          </div>
        </div>

        {/* Center: Countdown Timer */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          padding: "8px 18px",
          background: isTimeCritical ? "rgba(239, 68, 68, 0.15)" : "#1c1c1c",
          border: `1px solid ${isTimeCritical ? "var(--danger)" : "var(--border-strong)"}`,
          borderRadius: "var(--radius-full)",
          color: isTimeCritical ? "#f87171" : "#fff",
          fontWeight: 700,
          fontSize: "1.1rem",
          fontFamily: "var(--font-mono)",
        }}>
          <Clock size={18} color={isTimeCritical ? "var(--danger)" : "#c084fc"} />
          <span>{String(minutes).padStart(2, "0")}:{String(seconds).padStart(2, "0")}</span>
        </div>

        {/* Right: Submit Button */}
        <button
          onClick={() => {
            if (window.confirm("Are you sure you want to finish and submit your exam?")) {
              handleFinalSubmit();
            }
          }}
          disabled={isSubmitting}
          className="btn-primary"
          style={{ padding: "8px 18px", fontSize: "0.875rem" }}
        >
          <Send size={15} />
          {isSubmitting ? "Submitting..." : "Submit Exam"}
        </button>
      </header>

      {/* Main Examination Area */}
      <div style={{
        flex: 1,
        maxWidth: "1100px",
        width: "100%",
        margin: "0 auto",
        padding: "32px 24px 100px",
        display: "grid",
        gridTemplateColumns: "1fr 280px",
        gap: "30px",
      }}>
        {/* Left: Active Question Card */}
        <div>
          <div className="glass-panel" style={{ padding: "36px", marginBottom: "24px", background: "#1a1a1a" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
              <span className="badge badge-indigo">Question {currentQuestionIndex + 1}</span>
              <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                Worth: <strong style={{ color: "#c084fc" }}>{currentQ.marks || 1} Marks</strong>
              </span>
            </div>

            <h3 style={{ fontSize: "1.35rem", fontWeight: 700, lineHeight: 1.5, marginBottom: "28px", color: "#ffffff" }}>
              {currentQ.questionText}
            </h3>

            {/* Options List */}
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              {currentQ.options?.map((opt, optIdx) => {
                const isSelected = selectedAnswers[currentQ._id] === optIdx;
                return (
                  <div
                    key={optIdx}
                    onClick={() => handleSelectOption(optIdx)}
                    style={{
                      padding: "16px 20px",
                      background: isSelected ? "rgba(124, 58, 237, 0.16)" : "#161616",
                      border: `1.5px solid ${isSelected ? "var(--brand-primary)" : "var(--border-strong)"}`,
                      borderRadius: "var(--radius-lg)",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "14px",
                      transition: "all var(--transition-fast)",
                      boxShadow: isSelected ? "0 0 18px rgba(124, 58, 237, 0.28)" : "none",
                    }}
                  >
                    <div style={{
                      width: "32px",
                      height: "32px",
                      borderRadius: "50%",
                      background: isSelected ? "linear-gradient(135deg, #7c3aed 0%, #c026d3 100%)" : "#222222",
                      color: isSelected ? "#fff" : "var(--text-muted)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontWeight: 700,
                      fontSize: "0.9rem",
                    }}>
                      {String.fromCharCode(65 + optIdx)}
                    </div>
                    <span style={{ fontSize: "1rem", color: isSelected ? "#fff" : "var(--text-primary)" }}>
                      {opt}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Question Action Navigation Buttons */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <button
              onClick={() => setCurrentQuestionIndex((prev) => Math.max(0, prev - 1))}
              disabled={currentQuestionIndex === 0}
              className="btn-secondary"
              style={{ padding: "10px 18px", opacity: currentQuestionIndex === 0 ? 0.4 : 1 }}
            >
              <ChevronLeft size={16} />
              Previous
            </button>

            {currentQuestionIndex < questions.length - 1 ? (
              <button
                onClick={() => setCurrentQuestionIndex((prev) => Math.min(questions.length - 1, prev + 1))}
                className="btn-primary"
                style={{ padding: "10px 24px" }}
              >
                Save & Next
                <ChevronRight size={16} />
              </button>
            ) : (
              <button
                onClick={handleFinalSubmit}
                disabled={isSubmitting}
                className="btn-primary"
                style={{ padding: "10px 24px" }}
              >
                Finish & Submit
                <CheckCircle2 size={16} />
              </button>
            )}
          </div>
        </div>

        {/* Right: Question Palette / Progress */}
        <div>
          <div className="glass-panel" style={{ padding: "20px", background: "#1a1a1a" }}>
            <h4 style={{ fontSize: "0.95rem", fontWeight: 700, marginBottom: "14px", color: "#fff" }}>
              Question Palette
            </h4>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px" }}>
              {questions.map((q, idx) => {
                const isAnswered = selectedAnswers[q._id] !== undefined;
                const isCurrent = idx === currentQuestionIndex;

                let bg = "#181818";
                let borderColor = "var(--border-subtle)";
                let color = "var(--text-muted)";

                if (isCurrent) {
                  borderColor = "var(--brand-primary)";
                  color = "#fff";
                  bg = "rgba(124, 58, 237, 0.25)";
                } else if (isAnswered) {
                  bg = "rgba(52, 211, 153, 0.14)";
                  borderColor = "var(--success)";
                  color = "#34d399";
                }

                return (
                  <button
                    key={idx}
                    onClick={() => setCurrentQuestionIndex(idx)}
                    style={{
                      padding: "10px 0",
                      borderRadius: "var(--radius-sm)",
                      background: bg,
                      border: `1px solid ${borderColor}`,
                      color: color,
                      fontWeight: 700,
                      fontSize: "0.85rem",
                      cursor: "pointer",
                      transition: "all var(--transition-fast)",
                    }}
                  >
                    {idx + 1}
                  </button>
                );
              })}
            </div>

            <div style={{ marginTop: "20px", borderTop: "1px solid var(--border-subtle)", paddingTop: "14px", fontSize: "0.75rem", color: "var(--text-muted)", display: "flex", flexDirection: "column", gap: "6px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: "rgba(52, 211, 153, 0.25)", border: "1px solid var(--success)" }} />
                Answered ({Object.keys(selectedAnswers).length})
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: "#181818", border: "1px solid var(--border-subtle)" }} />
                Unanswered ({questions.length - Object.keys(selectedAnswers).length})
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Real-time Floating Proctor Camera */}
      <ProctorCamera
        socket={socket}
        attemptId={attemptId}
        sessionId={sessionId}
        strikes={strikes}
        maxStrikes={maxStrikes}
        onWarningIssued={handleWarningIssued}
        onForceTerminate={handleForceTerminate}
      />

      {/* Strike 1 & 2 Warning Modal */}
      {activeWarning && (
        <StrikeWarningModal
          warning={activeWarning}
          onAcknowledge={() => setActiveWarning(null)}
        />
      )}

      {/* Strike 3 Force Termination Overlay */}
      {isTerminated && (
        <TerminationModal
          reason={terminationReason}
          onExit={onExamTerminated}
        />
      )}
    </div>
  );
};
