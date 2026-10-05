import React, { useState } from "react";
import { Shield, Lock, Mail, User, AlertCircle, ArrowRight } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export const AuthPage = ({ onSuccess }) => {
  const [isRegister, setIsRegister] = useState(false);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("educator");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const { login, register } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (isRegister) {
        if (!name.trim()) throw new Error("Full name is required");
        await register(name, email, password, role);
      } else {
        await login(email, password);
      }
      onSuccess();
    } catch (err) {
      setError(err.message || "Authentication failed. Please check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      maxWidth: "480px",
      margin: "60px auto 100px",
      padding: "0 20px",
      position: "relative",
    }}>
      {/* Soft Ambient Purple Glow Behind Auth Card */}
      <div style={{
        position: "absolute",
        top: "20%",
        left: "50%",
        transform: "translateX(-50%)",
        width: "360px",
        height: "220px",
        background: "radial-gradient(circle, rgba(124, 58, 237, 0.2) 0%, rgba(192, 38, 211, 0.08) 50%, transparent 70%)",
        filter: "blur(50px)",
        pointerEvents: "none",
        zIndex: 0,
      }} />

      <div className="glass-panel" style={{ padding: "40px 36px", position: "relative", zIndex: 1, background: "#1a1a1a" }}>
        {/* Header Icon */}
        <div style={{ textAlign: "center", marginBottom: "28px" }}>
          <div style={{
            width: "52px",
            height: "52px",
            borderRadius: "14px",
            background: "linear-gradient(135deg, #7c3aed 0%, #c026d3 100%)",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 0 24px rgba(124, 58, 237, 0.45)",
            marginBottom: "16px",
          }}>
            <Shield size={28} color="#fff" />
          </div>

          <h2 style={{ fontSize: "1.75rem", fontWeight: 800, color: "#fff" }}>
            {isRegister ? "Create Account" : "Welcome Back"}
          </h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.925rem", marginTop: "6px" }}>
            {isRegister ? "Register as an Educator or Administrator" : "Sign in to access your proctoring dashboard"}
          </p>
        </div>

        {/* Tab Toggle */}
        <div style={{
          display: "flex",
          background: "#141414",
          padding: "4px",
          borderRadius: "var(--radius-md)",
          border: "1px solid var(--border-subtle)",
          marginBottom: "24px",
        }}>
          <button
            type="button"
            onClick={() => { setIsRegister(false); setError(""); }}
            style={{
              flex: 1,
              padding: "9px",
              borderRadius: "var(--radius-sm)",
              fontWeight: 600,
              fontSize: "0.875rem",
              background: !isRegister ? "linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)" : "transparent",
              color: !isRegister ? "#fff" : "var(--text-muted)",
              boxShadow: !isRegister ? "0 2px 10px rgba(124, 58, 237, 0.35)" : "none",
              transition: "all var(--transition-fast)",
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => { setIsRegister(true); setError(""); }}
            style={{
              flex: 1,
              padding: "9px",
              borderRadius: "var(--radius-sm)",
              fontWeight: 600,
              fontSize: "0.875rem",
              background: isRegister ? "linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)" : "transparent",
              color: isRegister ? "#fff" : "var(--text-muted)",
              boxShadow: isRegister ? "0 2px 10px rgba(124, 58, 237, 0.35)" : "none",
              transition: "all var(--transition-fast)",
            }}
          >
            Register
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div style={{
            background: "rgba(239, 68, 68, 0.12)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            borderRadius: "var(--radius-md)",
            padding: "12px 16px",
            color: "#fca5a5",
            fontSize: "0.875rem",
            display: "flex",
            alignItems: "center",
            gap: "10px",
            marginBottom: "20px",
          }}>
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
          {isRegister && (
            <div>
              <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "8px", color: "var(--text-secondary)" }}>
                Full Name
              </label>
              <div style={{ position: "relative" }}>
                <input
                  type="text"
                  required
                  placeholder="e.g. Dr. Alan Turing"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="form-input"
                  style={{ paddingLeft: "42px" }}
                />
                <User size={18} color="var(--text-muted)" style={{ position: "absolute", left: "14px", top: "14px" }} />
              </div>
            </div>
          )}

          <div>
            <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "8px", color: "var(--text-secondary)" }}>
              Email Address
            </label>
            <div style={{ position: "relative" }}>
              <input
                type="email"
                required
                placeholder="name@institution.edu"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="form-input"
                style={{ paddingLeft: "42px" }}
              />
              <Mail size={18} color="var(--text-muted)" style={{ position: "absolute", left: "14px", top: "14px" }} />
            </div>
          </div>

          <div>
            <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "8px", color: "var(--text-secondary)" }}>
              Password
            </label>
            <div style={{ position: "relative" }}>
              <input
                type="password"
                required
                placeholder="••••••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="form-input"
                style={{ paddingLeft: "42px" }}
              />
              <Lock size={18} color="var(--text-muted)" style={{ position: "absolute", left: "14px", top: "14px" }} />
            </div>
          </div>

          {isRegister && (
            <div>
              <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "8px", color: "var(--text-secondary)" }}>
                Account Role
              </label>
              <div style={{ display: "flex", gap: "10px" }}>
                <label style={{
                  flex: 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "6px",
                  padding: "10px",
                  background: role === "educator" ? "rgba(124, 58, 237, 0.16)" : "#161616",
                  border: `1px solid ${role === "educator" ? "#7c3aed" : "var(--border-strong)"}`,
                  borderRadius: "var(--radius-md)",
                  cursor: "pointer",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  color: role === "educator" ? "#c084fc" : "var(--text-secondary)",
                  transition: "all var(--transition-fast)",
                }}>
                  <input
                    type="radio"
                    name="role"
                    value="educator"
                    checked={role === "educator"}
                    onChange={(e) => setRole(e.target.value)}
                    style={{ display: "none" }}
                  />
                  Educator
                </label>

                <label style={{
                  flex: 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "6px",
                  padding: "10px",
                  background: role === "admin" ? "rgba(192, 38, 211, 0.16)" : "#161616",
                  border: `1px solid ${role === "admin" ? "#c026d3" : "var(--border-strong)"}`,
                  borderRadius: "var(--radius-md)",
                  cursor: "pointer",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  color: role === "admin" ? "#e879f9" : "var(--text-secondary)",
                  transition: "all var(--transition-fast)",
                }}>
                  <input
                    type="radio"
                    name="role"
                    value="admin"
                    checked={role === "admin"}
                    onChange={(e) => setRole(e.target.value)}
                    style={{ display: "none" }}
                  />
                  Admin
                </label>
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="btn-primary"
            style={{ width: "100%", padding: "12px", marginTop: "10px", fontSize: "0.95rem" }}
          >
            {loading ? "Processing..." : isRegister ? "Create Account" : "Sign In to Dashboard"}
            <ArrowRight size={18} />
          </button>
        </form>
      </div>
    </div>
  );
};
