import React, { useState, useEffect } from "react";
import { ArrowLeft, Users, ShieldAlert, Award, Clock, AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import { dashboardApi } from "../api/client";

export const TestAnalytics = ({ test, onBack }) => {
  const [attempts, setAttempts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedAttemptId, setSelectedAttemptId] = useState(null);
  const [attemptDetails, setAttemptDetails] = useState(null);
  const [proctoringAudit, setProctoringAudit] = useState(null);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("questions"); // "questions" | "audit"

  useEffect(() => {
    const fetchAttempts = async () => {
      try {
        setLoading(true);
        const res = await dashboardApi.getTestAttempts(test._id);
        if (res.success) {
          setAttempts(res.data.attempts || []);
        }
      } catch (err) {
        console.error("Failed to load attempts:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchAttempts();
  }, [test._id]);

  const handleSelectAttempt = async (attemptId) => {
    setSelectedAttemptId(attemptId);
    setDetailsLoading(true);
    try {
      const [detailRes, auditRes] = await Promise.all([
        dashboardApi.getAttemptDetails(attemptId),
        dashboardApi.getProctoringAudit(attemptId),
      ]);

      if (detailRes.success) setAttemptDetails(detailRes.data);
      if (auditRes.success) setProctoringAudit(auditRes.data);
    } catch (err) {
      console.error("Failed to load attempt details:", err);
    } finally {
      setDetailsLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "40px 24px 80px" }}>
      {/* Back Button & Header */}
      <button
        onClick={onBack}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "8px",
          color: "var(--text-secondary)",
          marginBottom: "20px",
          fontSize: "0.9rem",
          cursor: "pointer",
        }}
      >
        <ArrowLeft size={16} />
        Back to Assessments List
      </button>

      <div style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "16px",
        marginBottom: "32px",
        paddingBottom: "24px",
        borderBottom: "1px solid var(--border-subtle)",
      }}>
        <div>
          <span className="badge badge-indigo" style={{ marginBottom: "8px" }}>
            EXAM AUDIT REPORT
          </span>
          <h1 style={{ fontSize: "1.85rem", fontWeight: 800, color: "#fff" }}>{test.title}</h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginTop: "4px" }}>
            Shareable Code: <code style={{ color: "#c084fc", fontFamily: "var(--font-mono)" }}>{test.shareableLink}</code> • Duration: {test.duration} min
          </p>
        </div>

        <div style={{ display: "flex", gap: "16px" }}>
          <div className="glass-panel" style={{ padding: "12px 20px", textAlign: "center", background: "#1a1a1a" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>CANDIDATES</div>
            <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "#fff" }}>{attempts.length}</div>
          </div>
          <div className="glass-panel" style={{ padding: "12px 20px", textAlign: "center", background: "#1a1a1a" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 600 }}>TERMINATED</div>
            <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--danger)" }}>
              {attempts.filter((a) => a.status === "terminated").length}
            </div>
          </div>
        </div>
      </div>

      {/* Main Grid: Attempts List (Left) and Inspector (Right) */}
      <div style={{ display: "grid", gridTemplateColumns: selectedAttemptId ? "1fr 1.3fr" : "1fr", gap: "24px" }}>
        {/* Attempts Table */}
        <div className="glass-panel" style={{ padding: "20px", background: "#1a1a1a" }}>
          <h3 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "16px", color: "#fff" }}>
            Submissions ({attempts.length})
          </h3>

          {loading ? (
            <div style={{ padding: "30px", textAlign: "center", color: "var(--text-muted)" }}>
              Loading candidate submissions...
            </div>
          ) : attempts.length === 0 ? (
            <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>
              No candidates have taken this assessment yet.
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              {attempts.map((att) => {
                const isSelected = selectedAttemptId === att._id;
                return (
                  <div
                    key={att._id}
                    onClick={() => handleSelectAttempt(att._id)}
                    style={{
                      padding: "16px",
                      background: isSelected ? "rgba(124, 58, 237, 0.16)" : "#161616",
                      border: `1px solid ${isSelected ? "var(--brand-primary)" : "var(--border-strong)"}`,
                      borderRadius: "var(--radius-md)",
                      cursor: "pointer",
                      transition: "all var(--transition-fast)",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div>
                        <div style={{ fontWeight: 700, fontSize: "0.95rem", color: "#fff" }}>
                          {att.candidateId?.name || "Unknown Candidate"}
                        </div>
                        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                          {att.candidateId?.email}
                        </div>
                      </div>

                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: "1rem", fontWeight: 800, color: att.percentage >= 60 ? "var(--success)" : "#fbbf24" }}>
                          {att.score} / {att.totalMarks} ({att.percentage}%)
                        </div>
                        <span className={`badge ${
                          att.status === "submitted" ? "badge-success" : att.status === "terminated" ? "badge-danger" : "badge-warning"
                        }`} style={{ fontSize: "0.65rem" }}>
                          {att.status}
                        </span>
                      </div>
                    </div>

                    <div style={{ display: "flex", gap: "12px", marginTop: "10px", fontSize: "0.75rem", color: "var(--text-muted)" }}>
                      <span>Strikes: <strong style={{ color: att.strikes > 0 ? "var(--danger)" : "var(--success)" }}>{att.strikes}/3</strong></span>
                      <span>•</span>
                      <span>Violations: {att.anomalyCount || 0}</span>
                      <span>•</span>
                      <span>{new Date(att.startedAt).toLocaleTimeString()}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Detailed Inspector Panel */}
        {selectedAttemptId && (
          <div className="glass-panel" style={{ padding: "24px", background: "#1a1a1a" }}>
            {detailsLoading ? (
              <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>
                Loading assessment and AI audit details...
              </div>
            ) : attemptDetails ? (
              <div>
                {/* Header of Inspector */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "20px" }}>
                  <div>
                    <h3 style={{ fontSize: "1.2rem", fontWeight: 700, color: "#fff" }}>
                      {attemptDetails.attempt.candidate?.name}
                    </h3>
                    <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                      Score: <strong>{attemptDetails.attempt.score} / {attemptDetails.attempt.totalMarks}</strong> ({attemptDetails.attempt.percentage}%)
                    </p>
                  </div>

                  <span className={`badge ${
                    attemptDetails.attempt.status === "terminated" ? "badge-danger" : "badge-success"
                  }`}>
                    {attemptDetails.attempt.status}
                  </span>
                </div>

                {/* Tab Switcher */}
                <div style={{ display: "flex", gap: "10px", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "12px", marginBottom: "20px" }}>
                  <button
                    onClick={() => setActiveTab("questions")}
                    style={{
                      padding: "8px 16px",
                      borderRadius: "var(--radius-sm)",
                      fontWeight: 600,
                      fontSize: "0.85rem",
                      background: activeTab === "questions" ? "linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)" : "#1f1f1f",
                      color: activeTab === "questions" ? "#fff" : "var(--text-muted)",
                      border: "1px solid var(--border-subtle)",
                    }}
                  >
                    Question Responses ({attemptDetails.questions.length})
                  </button>

                  <button
                    onClick={() => setActiveTab("audit")}
                    style={{
                      padding: "8px 16px",
                      borderRadius: "var(--radius-sm)",
                      fontWeight: 600,
                      fontSize: "0.85rem",
                      background: activeTab === "audit" ? "linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)" : "#1f1f1f",
                      color: activeTab === "audit" ? "#fff" : "var(--text-muted)",
                      border: "1px solid var(--border-subtle)",
                      display: "flex",
                      alignItems: "center",
                      gap: "6px",
                    }}
                  >
                    <ShieldAlert size={14} />
                    AI Proctoring Audit ({proctoringAudit?.logs?.length || 0})
                  </button>
                </div>

                {/* Tab 1: Questions Breakdown */}
                {activeTab === "questions" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: "16px", maxHeight: "60vh", overflowY: "auto" }}>
                    {attemptDetails.questions.map((q, idx) => (
                      <div
                        key={idx}
                        style={{
                          background: "#151515",
                          border: `1px solid ${q.isCorrect ? "rgba(52, 211, 153, 0.3)" : "rgba(239, 68, 68, 0.3)"}`,
                          borderRadius: "var(--radius-md)",
                          padding: "16px",
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px" }}>
                          <span style={{ fontWeight: 700, fontSize: "0.9rem", color: "#fff" }}>
                            #{q.questionNumber}. {q.questionText}
                          </span>
                          <span className={`badge ${q.isCorrect ? "badge-success" : "badge-danger"}`} style={{ fontSize: "0.7rem" }}>
                            {q.marksAwarded} / {q.maxMarks} Marks
                          </span>
                        </div>

                        <div style={{ fontSize: "0.825rem", marginTop: "8px" }}>
                          <div style={{ color: q.isCorrect ? "#34d399" : "#fca5a5", marginBottom: "4px" }}>
                            <strong>Candidate Answer:</strong> {q.selectedOptionText ? q.selectedOptionText : "(Not Answered)"}
                          </div>
                          {!q.isCorrect && (
                            <div style={{ color: "#c084fc" }}>
                              <strong>Correct Answer:</strong> {q.correctOptionText}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Tab 2: Multimodal AI Proctoring Timeline */}
                {activeTab === "audit" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: "14px", maxHeight: "60vh", overflowY: "auto" }}>
                    <div style={{
                      background: "#151515",
                      border: "1px solid var(--border-strong)",
                      padding: "12px",
                      borderRadius: "var(--radius-md)",
                      fontSize: "0.85rem",
                      display: "flex",
                      justifyContent: "space-between",
                    }}>
                      <span>Strikes Enforced: <strong>{proctoringAudit?.session?.strikes || 0} / 3</strong></span>
                      <span>Total Anomalies Flagged: <strong>{proctoringAudit?.logs?.length || 0}</strong></span>
                    </div>

                    {proctoringAudit?.logs?.length === 0 ? (
                      <div style={{ padding: "30px", textAlign: "center", color: "var(--success)" }}>
                        <CheckCircle size={32} style={{ margin: "0 auto 8px" }} />
                        <p style={{ fontWeight: 600 }}>Clean Proctoring Record</p>
                        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                          No gaze, face, or audio anomalies were logged for this candidate.
                        </span>
                      </div>
                    ) : (
                      proctoringAudit?.logs?.map((log, idx) => (
                        <div
                          key={idx}
                          style={{
                            background: "#161616",
                            border: "1px solid var(--border-subtle)",
                            borderLeft: `4px solid ${log.strikeIssued ? "var(--danger)" : "var(--warning)"}`,
                            borderRadius: "var(--radius-sm)",
                            padding: "12px 14px",
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                            <span style={{ fontWeight: 700, fontSize: "0.85rem", color: "#fca5a5" }}>
                              {log.type.replace(/_/g, " ").toUpperCase()}
                            </span>
                            {log.strikeIssued && (
                              <span className="badge badge-danger" style={{ fontSize: "0.65rem" }}>
                                Strike {log.strikeNumber}
                              </span>
                            )}
                          </div>
                          <p style={{ fontSize: "0.85rem", color: "#e5e5e5" }}>{log.message}</p>
                          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "6px" }}>
                            Detected at: {new Date(log.detectedAt).toLocaleTimeString()}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
};
