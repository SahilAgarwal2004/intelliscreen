import React from "react";
import { Shield, LogOut, User, Layers } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export const Navbar = ({ currentView, setCurrentView }) => {
  const { user, isAuthenticated, logout } = useAuth();

  return (
    <header style={{
      position: "sticky",
      top: 0,
      zIndex: 100,
      background: "rgba(17, 17, 17, 0.88)",
      backdropFilter: "blur(20px)",
      WebkitBackdropFilter: "blur(20px)",
      borderBottom: "1px solid var(--border-subtle)",
      padding: "14px 28px",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
    }}>
      {/* Brand Logo */}
      <div 
        onClick={() => setCurrentView("landing")}
        style={{ display: "flex", alignItems: "center", gap: "10px", cursor: "pointer" }}
      >
        <div style={{
          width: "36px",
          height: "36px",
          borderRadius: "10px",
          background: "linear-gradient(135deg, #7c3aed 0%, #c026d3 100%)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: "0 0 18px rgba(124, 58, 237, 0.45)",
        }}>
          <Shield size={20} color="#fff" />
        </div>
        <div>
          <span style={{ fontWeight: 800, fontSize: "1.2rem", letterSpacing: "-0.02em" }}>
            Intelli<span className="text-gradient">Screen</span>
          </span>
          <span style={{
            fontSize: "0.65rem",
            background: "rgba(124, 58, 237, 0.15)",
            color: "#c084fc",
            border: "1px solid rgba(124, 58, 237, 0.3)",
            padding: "2px 6px",
            borderRadius: "4px",
            marginLeft: "8px",
            fontWeight: 700,
            textTransform: "uppercase",
          }}>AI PROCTOR</span>
        </div>
      </div>

      {/* Navigation Links */}
      <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
        {isAuthenticated ? (
          <>
            <button
              onClick={() => setCurrentView("dashboard")}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                padding: "8px 14px",
                borderRadius: "var(--radius-md)",
                background: currentView === "dashboard" ? "rgba(124, 58, 237, 0.16)" : "transparent",
                color: currentView === "dashboard" ? "#e879f9" : "var(--text-secondary)",
                border: currentView === "dashboard" ? "1px solid rgba(124, 58, 237, 0.3)" : "1px solid transparent",
                fontWeight: 600,
                fontSize: "0.9rem",
                transition: "all var(--transition-fast)",
              }}
            >
              <Layers size={16} />
              Tests & Exams
            </button>

            {/* User Role Badge */}
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "6px 14px",
              background: "#1c1c1c",
              border: "1px solid var(--border-strong)",
              borderRadius: "var(--radius-full)",
              fontSize: "0.85rem",
            }}>
              <User size={14} color="#c084fc" />
              <span style={{ fontWeight: 600, color: "#fff" }}>{user?.name}</span>
              <span className={`badge ${user?.role === "admin" ? "badge-danger" : "badge-indigo"}`} style={{ fontSize: "0.65rem", padding: "2px 6px" }}>
                {user?.role}
              </span>
            </div>

            {/* Logout */}
            <button
              onClick={() => {
                logout();
                setCurrentView("landing");
              }}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                padding: "8px 12px",
                color: "var(--text-muted)",
                fontSize: "0.85rem",
                borderRadius: "var(--radius-md)",
                transition: "color var(--transition-fast)",
              }}
              onMouseEnter={(e) => e.currentTarget.style.color = "#f87171"}
              onMouseLeave={(e) => e.currentTarget.style.color = "var(--text-muted)"}
            >
              <LogOut size={16} />
            </button>
          </>
        ) : (
          <>
            <button
              onClick={() => setCurrentView("auth")}
              className="btn-secondary"
              style={{ padding: "8px 16px", fontSize: "0.875rem" }}
            >
              Educator Login
            </button>
            <button
              onClick={() => setCurrentView("candidate_entry")}
              className="btn-primary"
              style={{ padding: "8px 18px", fontSize: "0.875rem" }}
            >
              Candidate Portal
            </button>
          </>
        )}
      </div>
    </header>
  );
};
