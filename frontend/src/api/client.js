const API_BASE = "";

export async function apiRequest(endpoint, options = {}) {
  const token = localStorage.getItem("intelliscreen_token");

  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers || {}),
  };

  const config = {
    ...options,
    headers,
  };

  const response = await fetch(endpoint, config);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const errorMsg = data?.message || `Request failed with status ${response.status}`;
    const error = new Error(errorMsg);
    error.status = response.status;
    error.data = data;
    throw error;
  }

  return data;
}

// ----------------------------------------------------
// Authentication API
// ----------------------------------------------------
export const authApi = {
  login: (email, password) =>
    apiRequest("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  register: (name, email, password, role = "educator") =>
    apiRequest("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ name, email, password, role }),
    }),

  getMe: () => apiRequest("/api/auth/me"),
};

// ----------------------------------------------------
// Test Management API (Admin / Educator)
// ----------------------------------------------------
export const testApi = {
  getMyTests: () => apiRequest("/api/tests"),

  getTestById: (testId) => apiRequest(`/api/tests/${testId}`),

  createTest: (payload) =>
    apiRequest("/api/tests", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  updateTest: (testId, payload) =>
    apiRequest(`/api/tests/${testId}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),

  deleteTest: (testId) =>
    apiRequest(`/api/tests/${testId}`, {
      method: "DELETE",
    }),

  toggleStatus: (testId) =>
    apiRequest(`/api/tests/${testId}/toggle-status`, {
      method: "PATCH",
    }),

  regenerateLink: (testId) =>
    apiRequest(`/api/tests/${testId}/regenerate-link`, {
      method: "POST",
    }),
};

// ----------------------------------------------------
// Candidate Assessment API
// ----------------------------------------------------
export const candidateApi = {
  getTestInfo: (shareableLink) =>
    apiRequest(`/api/candidate/test-info/${shareableLink}`),

  startAttempt: (payload) =>
    apiRequest("/api/candidate/start-attempt", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  saveAnswer: (attemptId, questionId, selectedOption) =>
    apiRequest("/api/candidate/answer", {
      method: "POST",
      body: JSON.stringify({ attemptId, questionId, selectedOption }),
    }),

  submitTest: (attemptId) =>
    apiRequest("/api/candidate/submit", {
      method: "POST",
      body: JSON.stringify({ attemptId }),
    }),

  getAttemptStatus: (attemptId) =>
    apiRequest(`/api/candidate/attempt/${attemptId}`),
};

// ----------------------------------------------------
// Dashboard & Audit API
// ----------------------------------------------------
export const dashboardApi = {
  getTestAttempts: (testId) =>
    apiRequest(`/api/dashboard/tests/${testId}/attempts`),

  getAttemptDetails: (attemptId) =>
    apiRequest(`/api/dashboard/attempts/${attemptId}`),

  getProctoringAudit: (attemptId) =>
    apiRequest(`/api/dashboard/attempts/${attemptId}/proctoring`),
};
