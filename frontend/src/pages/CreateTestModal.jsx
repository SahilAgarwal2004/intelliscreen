import React, { useState } from "react";
import { X, Plus, Trash2, CheckCircle2, Clock, Award } from "lucide-react";
import { testApi } from "../api/client";

export const CreateTestModal = ({ onClose, onSuccess }) => {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [duration, setDuration] = useState(30);
  const [maxAttempts, setMaxAttempts] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [questions, setQuestions] = useState([
    {
      questionText: "",
      options: ["", "", "", ""],
      correctAnswer: 0,
      marks: 1,
    },
  ]);

  const handleAddQuestion = () => {
    setQuestions([
      ...questions,
      {
        questionText: "",
        options: ["", "", "", ""],
        correctAnswer: 0,
        marks: 1,
      },
    ]);
  };

  const handleRemoveQuestion = (idx) => {
    if (questions.length === 1) return;
    setQuestions(questions.filter((_, i) => i !== idx));
  };

  const handleQuestionTextChange = (idx, text) => {
    const updated = [...questions];
    updated[idx].questionText = text;
    setQuestions(updated);
  };

  const handleOptionChange = (qIdx, optIdx, text) => {
    const updated = [...questions];
    updated[qIdx].options[optIdx] = text;
    setQuestions(updated);
  };

  const handleCorrectAnswerChange = (qIdx, optIdx) => {
    const updated = [...questions];
    updated[qIdx].correctAnswer = optIdx;
    setQuestions(updated);
  };

  const handleMarksChange = (qIdx, marks) => {
    const updated = [...questions];
    updated[qIdx].marks = Math.max(1, Number(marks) || 1);
    setQuestions(updated);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    // Validate questions
    for (let i = 0; i < questions.length; i++) {
      const q = questions[i];
      if (!q.questionText.trim()) {
        setError(`Question #${i + 1} is missing question text.`);
        return;
      }
      for (let j = 0; j < q.options.length; j++) {
        if (!q.options[j].trim()) {
          setError(`Question #${i + 1}, Option ${String.fromCharCode(65 + j)} cannot be empty.`);
          return;
        }
      }
    }

    setLoading(true);
    try {
      const payload = {
        title: title.trim(),
        description: description.trim(),
        duration: Number(duration),
        maxAttempts: Number(maxAttempts),
        questions,
      };

      const res = await testApi.createTest(payload);
      if (res.success) {
        onSuccess(res.data);
        onClose();
      }
    } catch (err) {
      setError(err.message || "Failed to create assessment");
    } finally {
      setLoading(false);
    }
  };

  const totalMarks = questions.reduce((acc, q) => acc + (q.marks || 1), 0);

  return (
    <div style={{
      position: "fixed",
      inset: 0,
      zIndex: 200,
      background: "rgba(10, 10, 10, 0.85)",
      backdropFilter: "blur(14px)",
      WebkitBackdropFilter: "blur(14px)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "20px",
    }}>
      <div className="glass-panel" style={{
        maxWidth: "800px",
        width: "100%",
        maxHeight: "90vh",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        background: "#1a1a1a",
        border: "1px solid var(--border-strong)",
      }}>
        {/* Modal Header */}
        <div style={{
          padding: "20px 28px",
          borderBottom: "1px solid var(--border-subtle)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "#181818",
        }}>
          <div>
            <h2 style={{ fontSize: "1.35rem", fontWeight: 700, color: "#fff" }}>Create Proctored MCQ Assessment</h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "2px" }}>
              Configure questions, answers, and time parameters
            </p>
          </div>
          <button
            onClick={onClose}
            style={{
              padding: "6px",
              borderRadius: "50%",
              color: "var(--text-muted)",
              transition: "all var(--transition-fast)",
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Scrollable Form Body */}
        <form onSubmit={handleSubmit} style={{ overflowY: "auto", padding: "28px", flex: 1 }}>
          {error && (
            <div style={{
              background: "rgba(239, 68, 68, 0.12)",
              border: "1px solid rgba(239, 68, 68, 0.3)",
              borderRadius: "var(--radius-md)",
              padding: "12px",
              color: "#fca5a5",
              fontSize: "0.875rem",
              marginBottom: "20px",
            }}>
              {error}
            </div>
          )}

          {/* Test Metadata */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "20px" }}>
            <div style={{ gridColumn: "span 2" }}>
              <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "6px", color: "var(--text-secondary)" }}>
                Assessment Title
              </label>
              <input
                type="text"
                required
                placeholder="e.g. Distributed Systems & Concurrency Midterm"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="form-input"
              />
            </div>

            <div style={{ gridColumn: "span 2" }}>
              <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "6px", color: "var(--text-secondary)" }}>
                Instructions / Description (Optional)
              </label>
              <textarea
                placeholder="Brief instructions for candidates regarding proctoring rules..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="form-input"
                rows={2}
                style={{ resize: "vertical" }}
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "6px", color: "var(--text-secondary)" }}>
                Duration (Minutes)
              </label>
              <input
                type="number"
                min={1}
                required
                value={duration}
                onChange={(e) => setDuration(e.target.value)}
                className="form-input"
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "6px", color: "var(--text-secondary)" }}>
                Max Allowed Attempts per Candidate
              </label>
              <input
                type="number"
                min={1}
                max={5}
                required
                value={maxAttempts}
                onChange={(e) => setMaxAttempts(e.target.value)}
                className="form-input"
              />
            </div>
          </div>

          {/* Questions Header */}
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            margin: "24px 0 16px",
            paddingBottom: "10px",
            borderBottom: "1px solid var(--border-subtle)",
          }}>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#fff" }}>
              Questions ({questions.length}) — Total Marks: <span style={{ color: "#c084fc" }}>{totalMarks}</span>
            </h3>
            <button
              type="button"
              onClick={handleAddQuestion}
              className="btn-secondary"
              style={{ padding: "6px 14px", fontSize: "0.85rem" }}
            >
              <Plus size={15} />
              Add Question
            </button>
          </div>

          {/* Questions List */}
          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            {questions.map((q, qIdx) => (
              <div
                key={qIdx}
                style={{
                  background: "#151515",
                  border: "1px solid var(--border-strong)",
                  borderRadius: "var(--radius-lg)",
                  padding: "20px",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
                  <span style={{ fontWeight: 700, fontSize: "0.95rem", color: "#c084fc" }}>
                    Question #{qIdx + 1}
                  </span>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                      <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Marks:</span>
                      <input
                        type="number"
                        min={1}
                        value={q.marks}
                        onChange={(e) => handleMarksChange(qIdx, e.target.value)}
                        style={{
                          width: "50px",
                          padding: "4px 8px",
                          background: "#1c1c1c",
                          border: "1px solid var(--border-strong)",
                          borderRadius: "var(--radius-sm)",
                          color: "#fff",
                          textAlign: "center",
                        }}
                      />
                    </div>
                    {questions.length > 1 && (
                      <button
                        type="button"
                        onClick={() => handleRemoveQuestion(qIdx)}
                        style={{ color: "#ef4444", padding: "4px" }}
                        title="Delete Question"
                      >
                        <Trash2 size={16} />
                      </button>
                    )}
                  </div>
                </div>

                {/* Question Text */}
                <input
                  type="text"
                  required
                  placeholder="Enter multiple-choice question prompt here..."
                  value={q.questionText}
                  onChange={(e) => handleQuestionTextChange(qIdx, e.target.value)}
                  className="form-input"
                  style={{ marginBottom: "14px" }}
                />

                {/* Options List */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                  {q.options.map((opt, optIdx) => (
                    <div
                      key={optIdx}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                        background: q.correctAnswer === optIdx ? "rgba(52, 211, 153, 0.12)" : "#1a1a1a",
                        border: `1px solid ${q.correctAnswer === optIdx ? "var(--success)" : "var(--border-strong)"}`,
                        borderRadius: "var(--radius-md)",
                        padding: "6px 10px",
                        transition: "all var(--transition-fast)",
                      }}
                    >
                      <input
                        type="radio"
                        name={`correct_answer_${qIdx}`}
                        checked={q.correctAnswer === optIdx}
                        onChange={() => handleCorrectAnswerChange(qIdx, optIdx)}
                        title="Mark as correct answer"
                        style={{ accentColor: "var(--success)", cursor: "pointer" }}
                      />
                      <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--text-muted)" }}>
                        {String.fromCharCode(65 + optIdx)}.
                      </span>
                      <input
                        type="text"
                        required
                        placeholder={`Option ${String.fromCharCode(65 + optIdx)}`}
                        value={opt}
                        onChange={(e) => handleOptionChange(qIdx, optIdx, e.target.value)}
                        style={{
                          flex: 1,
                          background: "transparent",
                          border: "none",
                          color: "#fff",
                          fontSize: "0.875rem",
                          outline: "none",
                        }}
                      />
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "8px" }}>
                  💡 Click the radio button next to an option to mark it as the correct answer.
                </div>
              </div>
            ))}
          </div>

          {/* Modal Actions */}
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            gap: "12px",
            marginTop: "30px",
            paddingTop: "16px",
            borderTop: "1px solid var(--border-subtle)",
          }}>
            <button type="button" onClick={onClose} className="btn-secondary">
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="btn-primary"
              style={{ minWidth: "160px" }}
            >
              {loading ? "Publishing..." : "Publish Assessment"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
