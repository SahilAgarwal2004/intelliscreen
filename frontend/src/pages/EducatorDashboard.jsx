import React, { useState, useEffect } from "react";
import { Plus, Copy, Check, BarChart3, RefreshCw, Trash2, Clock, Users, Sparkles } from "lucide-react";
import { testApi } from "../api/client";
import { CreateTestModal } from "./CreateTestModal";

export const EducatorDashboard = ({ onSelectTestForAnalytics }) => {
  const [tests, setTests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [copiedLink, setCopiedLink] = useState(null);
  const [actionMessage, setActionMessage] = useState("");

  const fetchTests = async () => {
    try {
      setLoading(true);
      const res = await testApi.getMyTests();
      if (res.success) {
        setTests(res.data || []);
      }
    } catch (err) {
      console.error("Failed to load tests:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTests();
  }, []);

  const handleCopyLink = (shareableLink) => {
    const fullUrl = `${window.location.origin}/test/${shareableLink}`;
    navigator.clipboard.writeText(fullUrl);
    setCopiedLink(shareableLink);
    setTimeout(() => setCopiedLink(null), 2500);
  };

  const handleToggleStatus = async (testId) => {
    try {
      const res = await testApi.toggleStatus(testId);
      if (res.success) {
        setTests(tests.map((t) => (t._id === testId ? { ...t, isActive: res.data.isActive } : t)));
      }
    } catch (err) {
      console.error("Toggle status error:", err);
    }
  };

  const handleRegenerateLink = async (testId) => {
    if (!window.confirm("Regenerating this link will invalidate the old link for candidates. Proceed?")) return;
    try {
      const res = await testApi.regenerateLink(testId);
      if (res.success) {
        setTests(tests.map((t) => (t._id === testId ? { ...t, shareableLink: res.data.shareableLink } : t)));
        setActionMessage("Test link regenerated successfully!");
        setTimeout(() => setActionMessage(""), 3000);
      }
    } catch (err) {
      console.error("Regenerate link error:", err);
    }
  };

  const handleDeleteTest = async (testId) => {
    if (!window.confirm("Are you sure you want to delete this test and its candidate attempts?")) return;
    try {
      const res = await testApi.deleteTest(testId);
      if (res.success) {
        setTests(tests.filter((t) => t._id !== testId));
      }
    } catch (err) {
      console.error("Delete test error:", err);
    }
  };

  const totalAttempts = tests.reduce((sum, t) => sum + (t.attemptsCount || 0), 0);
  const activeTests = tests.filter((t) => t.isActive).length;

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "40px 24px 80px" }}>
      {/* Dashboard Top Header */}
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "20px",
        marginBottom: "36px",
      }}>
        <div>
          <h1 style={{ fontSize: "2rem", fontWeight: 800, color: "#fff" }}>Educator Control Center</h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", marginTop: "4px" }}>
            Create tests, issue unique proctored exam links, and inspect multimodal AI audits.
          </p>
        </div>

        <button
          onClick={() => setShowCreateModal(true)}
          className="btn-primary"
          style={{ padding: "12px 22px" }}
        >
          <Plus size={18} />
          Create New Assessment
        </button>
      </div>

      {actionMessage && (
        <div style={{
          background: "rgba(52, 211, 153, 0.12)",
          border: "1px solid var(--success)",
          color: "#34d399",
          padding: "12px 18px",
          borderRadius: "var(--radius-md)",
          marginBottom: "24px",
          fontSize: "0.9rem",
        }}>
          {actionMessage}
        </div>
      )}

      {/* Metric Cards Row */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
        gap: "20px",
        marginBottom: "40px",
      }}>
        <div className="glass-panel" style={{ padding: "24px", background: "linear-gradient(180deg, #1d1d1d 0%, #171717 100%)" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ color: "var(--text-secondary)", fontSize: "0.85rem", fontWeight: 600 }}>TOTAL ASSESSMENTS</span>
            <span className="badge badge-indigo">Overview</span>
          </div>
          <div style={{ fontSize: "2.25rem", fontWeight: 800, marginTop: "10px", color: "#fff" }}>
            {tests.length}
          </div>
          <div style={{ fontSize: "0.825rem", color: "var(--text-muted)", marginTop: "6px" }}>
            {activeTests} tests currently accepting candidates
          </div>
        </div>

        <div className="glass-panel" style={{ padding: "24px", background: "linear-gradient(180deg, #1d1d1d 0%, #171717 100%)" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ color: "var(--text-secondary)", fontSize: "0.85rem", fontWeight: 600 }}>CANDIDATE SUBMISSIONS</span>
            <span className="badge badge-success">Evaluated</span>
          </div>
          <div style={{ fontSize: "2.25rem", fontWeight: 800, marginTop: "10px", color: "var(--success)" }}>
            {totalAttempts}
          </div>
          <div style={{ fontSize: "0.825rem", color: "var(--text-muted)", marginTop: "6px" }}>
            With full AI suspicion scoring & proctoring audits
          </div>
        </div>

        <div className="glass-panel" style={{ padding: "24px", background: "linear-gradient(180deg, #1d1d1d 0%, #171717 100%)" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ color: "var(--text-secondary)", fontSize: "0.85rem", fontWeight: 600 }}>PROCTORING SENSORS</span>
            <span className="badge badge-magenta">Active Engine</span>
          </div>
          <div style={{ fontSize: "1.2rem", fontWeight: 700, marginTop: "12px", display: "flex", alignItems: "center", gap: "8px", color: "#fff" }}>
            <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--success)" }} />
            Vision & Voice AI Ready
          </div>
          <div style={{ fontSize: "0.825rem", color: "var(--text-muted)", marginTop: "10px" }}>
            3-Strike Server-Side Enforcement Enabled
          </div>
        </div>
      </div>

      {/* Tests Table / Cards */}
      <h2 style={{ fontSize: "1.35rem", fontWeight: 700, marginBottom: "20px", color: "#fff" }}>Your Assessments</h2>

      {loading ? (
        <div style={{ textAlign: "center", padding: "60px", color: "var(--text-muted)" }}>
          Loading your assessments...
        </div>
      ) : tests.length === 0 ? (
        <div className="glass-panel" style={{ textAlign: "center", padding: "60px 20px" }}>
          <Sparkles size={40} color="#c084fc" style={{ margin: "0 auto 16px" }} />
          <h3 style={{ fontSize: "1.25rem", fontWeight: 700, color: "#fff" }}>No Assessments Created Yet</h3>
          <p style={{ color: "var(--text-secondary)", maxWidth: "440px", margin: "8px auto 24px" }}>
            Create your first MCQ test to generate a unique proctored link for candidates.
          </p>
          <button onClick={() => setShowCreateModal(true)} className="btn-primary">
            <Plus size={16} />
            Create Assessment Now
          </button>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {tests.map((test) => (
            <div
              key={test._id}
              className="glass-panel"
              style={{
                padding: "24px",
                display: "flex",
                flexWrap: "wrap",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "20px",
                background: "#1a1a1a",
              }}
            >
              <div style={{ flex: "1 1 320px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
                  <h3 style={{ fontSize: "1.15rem", fontWeight: 700, color: "#fff" }}>{test.title}</h3>
                  <span className={`badge ${test.isActive ? "badge-success" : "badge-danger"}`}>
                    {test.isActive ? "ACTIVE" : "INACTIVE"}
                  </span>
                </div>

                <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem", marginBottom: "12px" }}>
                  {test.description || "No description provided."}
                </p>

                <div style={{ display: "flex", flexWrap: "wrap", gap: "16px", fontSize: "0.825rem", color: "var(--text-muted)" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                    <Clock size={14} /> {test.duration} Minutes
                  </span>
                  <span>•</span>
                  <span>{test.totalQuestions || test.questions?.length || 0} Questions</span>
                  <span>•</span>
                  <span>{test.totalMarks || 0} Total Marks</span>
                  <span>•</span>
                  <span style={{ display: "flex", alignItems: "center", gap: "5px", color: "#c084fc" }}>
                    <Users size={14} /> {test.attemptsCount || 0} Submissions
                  </span>
                </div>
              </div>

              {/* Action Controls & Unique Link */}
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "10px" }}>
                {/* Copy Link Button */}
                <button
                  onClick={() => handleCopyLink(test.shareableLink)}
                  className="btn-secondary"
                  style={{
                    padding: "8px 14px",
                    fontSize: "0.85rem",
                    borderColor: copiedLink === test.shareableLink ? "var(--success)" : undefined,
                    color: copiedLink === test.shareableLink ? "var(--success)" : undefined,
                  }}
                  title="Copy shareable link for candidates"
                >
                  {copiedLink === test.shareableLink ? (
                    <>
                      <Check size={15} />
                      Link Copied!
                    </>
                  ) : (
                    <>
                      <Copy size={15} />
                      Code: <code style={{ fontFamily: "var(--font-mono)", color: "#c084fc" }}>{test.shareableLink}</code>
                    </>
                  )}
                </button>

                {/* Regenerate Link */}
                <button
                  onClick={() => handleRegenerateLink(test._id)}
                  className="btn-secondary"
                  style={{ padding: "8px 12px" }}
                  title="Regenerate unique test link"
                >
                  <RefreshCw size={15} />
                </button>

                {/* Toggle Active Status */}
                <button
                  onClick={() => handleToggleStatus(test._id)}
                  className="btn-secondary"
                  style={{ padding: "8px 14px", fontSize: "0.85rem" }}
                >
                  {test.isActive ? "Deactivate" : "Activate"}
                </button>

                {/* View Submissions & Analytics */}
                <button
                  onClick={() => onSelectTestForAnalytics(test)}
                  className="btn-primary"
                  style={{ padding: "8px 16px", fontSize: "0.85rem" }}
                >
                  <BarChart3 size={15} />
                  Analytics & Audit
                </button>

                {/* Delete */}
                <button
                  onClick={() => handleDeleteTest(test._id)}
                  style={{
                    padding: "8px 10px",
                    color: "var(--text-muted)",
                    borderRadius: "var(--radius-md)",
                    transition: "color var(--transition-fast)",
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.color = "#ef4444"}
                  onMouseLeave={(e) => e.currentTarget.style.color = "var(--text-muted)"}
                  title="Delete assessment"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {showCreateModal && (
        <CreateTestModal
          onClose={() => setShowCreateModal(false)}
          onSuccess={(newTest) => {
            setTests([newTest, ...tests]);
            setActionMessage(`Assessment "${newTest.title}" published successfully!`);
            setTimeout(() => setActionMessage(""), 3500);
          }}
        />
      )}
    </div>
  );
};
