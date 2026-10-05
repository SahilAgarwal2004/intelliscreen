import React, { useState, useEffect } from "react";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { Navbar } from "./components/Navbar";
import { LandingPage } from "./pages/LandingPage";
import { AuthPage } from "./pages/AuthPage";
import { EducatorDashboard } from "./pages/EducatorDashboard";
import { TestAnalytics } from "./pages/TestAnalytics";
import { CandidatePortal } from "./pages/CandidatePortal";
import { ExamSession } from "./pages/ExamSession";
import { ExamSubmitted } from "./pages/ExamSubmitted";

function MainApp() {
  const { isAuthenticated, loading } = useAuth();
  const [currentView, setCurrentView] = useState("landing");
  const [selectedTestForAnalytics, setSelectedTestForAnalytics] = useState(null);
  const [candidateShareableLink, setCandidateShareableLink] = useState("");
  const [activeExamData, setActiveExamData] = useState(null);
  const [examResult, setExamResult] = useState(null);

  // Check URL pathname on mount (e.g. /test/a19dbcf7cbec)
  useEffect(() => {
    const path = window.location.pathname;
    if (path.startsWith("/test/")) {
      const code = path.replace("/test/", "").trim();
      if (code) {
        setCandidateShareableLink(code);
        setCurrentView("candidate_entry");
      }
    }
  }, []);

  if (loading) {
    return (
      <div style={{
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--bg-primary)",
        color: "var(--text-muted)",
        fontSize: "1.1rem",
      }}>
        Initializing IntelliScreen...
      </div>
    );
  }

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Navbar is hidden during live exam session to provide distraction-free environment */}
      {currentView !== "exam_session" && (
        <Navbar currentView={currentView} setCurrentView={setCurrentView} />
      )}

      <main style={{ flex: 1 }}>
        {currentView === "landing" && (
          <LandingPage
            onGoToAuth={() => setCurrentView(isAuthenticated ? "dashboard" : "auth")}
            onGoToCandidate={() => setCurrentView("candidate_entry")}
            onTestCodeSubmit={(code) => {
              setCandidateShareableLink(code);
              setCurrentView("candidate_entry");
            }}
          />
        )}

        {currentView === "auth" && (
          <AuthPage onSuccess={() => setCurrentView("dashboard")} />
        )}

        {currentView === "dashboard" && (
          <EducatorDashboard
            onSelectTestForAnalytics={(test) => {
              setSelectedTestForAnalytics(test);
              setCurrentView("analytics");
            }}
          />
        )}

        {currentView === "analytics" && selectedTestForAnalytics && (
          <TestAnalytics
            test={selectedTestForAnalytics}
            onBack={() => {
              setSelectedTestForAnalytics(null);
              setCurrentView("dashboard");
            }}
          />
        )}

        {currentView === "candidate_entry" && (
          <CandidatePortal
            initialShareableLink={candidateShareableLink}
            onStartExam={(examPayload) => {
              setActiveExamData(examPayload);
              setCurrentView("exam_session");
            }}
          />
        )}

        {currentView === "exam_session" && activeExamData && (
          <ExamSession
            examData={activeExamData}
            onExamFinished={(resultPayload) => {
              setExamResult(resultPayload);
              setCurrentView("exam_submitted");
            }}
            onExamTerminated={() => {
              setActiveExamData(null);
              setCurrentView("landing");
            }}
          />
        )}

        {currentView === "exam_submitted" && (
          <ExamSubmitted
            result={examResult}
            onHome={() => {
              setActiveExamData(null);
              setExamResult(null);
              setCurrentView("landing");
            }}
          />
        )}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <MainApp />
    </AuthProvider>
  );
}
