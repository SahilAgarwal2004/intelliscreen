import http from "http";
import ioClient from "socket.io-client";
import dotenv from "dotenv";
import connectDB from "./src/db/index.js";
import app from "./src/app.js";
import { initializeSocket } from "./src/socket/proctoring.socket.js";

dotenv.config({ path: "./.env" });

const TEST_PORT = 5055;
const BASE_URL = `http://localhost:${TEST_PORT}`;

// Pretty colored logging helpers
const colors = {
  reset: "\x1b[0m",
  green: "\x1b[32m",
  red: "\x1b[31m",
  cyan: "\x1b[36m",
  yellow: "\x1b[33m",
  bold: "\x1b[1m",
};

let passedCount = 0;
let failedCount = 0;

const assert = (condition, testName, details = "") => {
  if (condition) {
    passedCount++;
    console.log(`  ${colors.green}✔ PASS:${colors.reset} ${testName}`);
  } else {
    failedCount++;
    console.error(`  ${colors.red}✖ FAIL:${colors.reset} ${testName} ${details ? `(${details})` : ""}`);
  }
};

const req = async (endpoint, options = {}) => {
  const url = `${BASE_URL}${endpoint}`;
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  const response = await fetch(url, {
    ...options,
    headers,
  });
  const data = await response.json();
  return { status: response.status, body: data };
};

async function runTestSuite() {
  console.log(`\n${colors.bold}${colors.cyan}====================================================${colors.reset}`);
  console.log(`${colors.bold}${colors.cyan}  IntelliScreen Full End-to-End API Test Suite     ${colors.reset}`);
  console.log(`${colors.bold}${colors.cyan}====================================================\n${colors.reset}`);

  // 1. Start Server on TEST_PORT
  await connectDB();
  const server = http.createServer(app);
  initializeSocket(server);

  await new Promise((resolve) => {
    server.listen(TEST_PORT, () => {
      console.log(`${colors.yellow}► Test server running on port ${TEST_PORT}${colors.reset}\n`);
      resolve();
    });
  });

  const timestamp = Date.now();
  let adminToken = "";
  let educatorToken = "";
  let createdTestId = "";
  let shareableLink = "";
  let candidateAttemptId = "";
  let candidateSessionId = "";
  let secondAttemptId = "";
  let secondSessionId = "";

  try {
    // ----------------------------------------------------
    // TEST GROUP 1: Health & System Endpoints
    // ----------------------------------------------------
    console.log(`${colors.bold}1. Health & System Check${colors.reset}`);
    const healthRes = await req("/health");
    assert(healthRes.status === 200, "GET /health returns 200 OK");
    assert(healthRes.body.status === "ok", "GET /health status is 'ok'");

    // ----------------------------------------------------
    // TEST GROUP 2: Authentication & Authorization
    // ----------------------------------------------------
    console.log(`\n${colors.bold}2. Authentication & Authorization${colors.reset}`);
    
    // Register Admin
    const adminEmail = `admin_${timestamp}@intelliscreen.com`;
    const regAdminRes = await req("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({
        name: "Admin User",
        email: adminEmail,
        password: "SecretPassword123!",
        role: "admin",
      }),
    });
    assert(regAdminRes.status === 201, "POST /api/auth/register (Admin) returns 201");
    assert(Boolean(regAdminRes.body.data.token), "Registration returns JWT access token");
    assert(regAdminRes.body.data.user.password === undefined, "Password is not leaked in user payload");
    adminToken = regAdminRes.body.data.token;

    // Register Educator
    const educatorEmail = `educator_${timestamp}@intelliscreen.com`;
    const regEduRes = await req("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({
        name: "Professor Turing",
        email: educatorEmail,
        password: "SecretPassword123!",
        role: "educator",
      }),
    });
    assert(regEduRes.status === 201, "POST /api/auth/register (Educator) returns 201");
    educatorToken = regEduRes.body.data.token;

    // Reject duplicate email
    const dupRes = await req("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({
        name: "Duplicate User",
        email: adminEmail,
        password: "password123",
      }),
    });
    assert(dupRes.status === 409, "POST /api/auth/register rejects duplicate email (409 Conflict)");

    // Test Invalid Login
    const badLogin = await req("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email: adminEmail, password: "wrongpassword" }),
    });
    assert(badLogin.status === 401, "POST /api/auth/login rejects wrong password (401)");

    // Test Valid Login
    const goodLogin = await req("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email: adminEmail, password: "SecretPassword123!" }),
    });
    assert(goodLogin.status === 200, "POST /api/auth/login succeeds with valid credentials (200)");
    assert(Boolean(goodLogin.body.data.token), "Login returns valid JWT token");

    // Profile verification
    const unauthMe = await req("/api/auth/me");
    assert(unauthMe.status === 401, "GET /api/auth/me rejects unauthenticated request (401)");

    const authMe = await req("/api/auth/me", {
      headers: { Authorization: `Bearer ${adminToken}` },
    });
    assert(authMe.status === 200, "GET /api/auth/me returns 200 with Bearer token");
    assert(authMe.body.data.email === adminEmail, "GET /api/auth/me returns correct user data");

    // ----------------------------------------------------
    // TEST GROUP 3: Test Creation & Management (Admin / Educator)
    // ----------------------------------------------------
    console.log(`\n${colors.bold}3. Test Management (Admin/Educator)${colors.reset}`);

    // Create a new test
    const createTestRes = await req("/api/tests", {
      method: "POST",
      headers: { Authorization: `Bearer ${educatorToken}` },
      body: JSON.stringify({
        title: "Algorithms & Data Structures Proctored Assessment",
        description: "Comprehensive exam testing Trees, Graphs, and Dynamic Programming",
        duration: 45,
        maxAttempts: 2,
        questions: [
          {
            questionText: "What is the worst-case time complexity of QuickSort?",
            options: ["O(n log n)", "O(n^2)", "O(n)", "O(log n)"],
            correctAnswer: 1, // "O(n^2)"
            marks: 2,
          },
          {
            questionText: "Which data structure operates on a First-In-First-Out (FIFO) basis?",
            options: ["Stack", "Queue", "Binary Tree", "Heap"],
            correctAnswer: 1, // "Queue"
            marks: 1,
          },
          {
            questionText: "In a min-heap, where is the smallest element located?",
            options: ["At the root", "At any leaf node", "At the last index", "In the middle"],
            correctAnswer: 0, // "At the root"
            marks: 2,
          },
        ],
      }),
    });
    assert(createTestRes.status === 201, "POST /api/tests creates a test (201 Created)");
    assert(Boolean(createTestRes.body.data.shareableLink), "Generated unique shareableLink");
    createdTestId = createTestRes.body.data._id;
    shareableLink = createTestRes.body.data.shareableLink;

    // Get my tests
    const getTestsRes = await req("/api/tests", {
      headers: { Authorization: `Bearer ${educatorToken}` },
    });
    assert(getTestsRes.status === 200, "GET /api/tests lists educator's tests");
    assert(getTestsRes.body.data.length >= 1, "Educator sees their created test");

    // Get single test by ID
    const singleTestRes = await req(`/api/tests/${createdTestId}`, {
      headers: { Authorization: `Bearer ${educatorToken}` },
    });
    assert(singleTestRes.status === 200, "GET /api/tests/:testId returns test details");
    assert(singleTestRes.body.data.totalMarks === 5, "Total marks computed accurately (5 marks)");

    // Update test
    const updateTestRes = await req(`/api/tests/${createdTestId}`, {
      method: "PUT",
      headers: { Authorization: `Bearer ${educatorToken}` },
      body: JSON.stringify({
        duration: 50,
      }),
    });
    assert(updateTestRes.status === 200, "PUT /api/tests/:testId updates duration");
    assert(updateTestRes.body.data.duration === 50, "Updated duration is 50 minutes");

    // Toggle test status
    const toggleRes = await req(`/api/tests/${createdTestId}/toggle-status`, {
      method: "PATCH",
      headers: { Authorization: `Bearer ${educatorToken}` },
    });
    assert(toggleRes.status === 200, "PATCH /api/tests/:testId/toggle-status works");
    assert(toggleRes.body.data.isActive === false, "Test is now toggled to inactive");

    // Toggle back to active
    await req(`/api/tests/${createdTestId}/toggle-status`, {
      method: "PATCH",
      headers: { Authorization: `Bearer ${educatorToken}` },
    });

    // ----------------------------------------------------
    // TEST GROUP 4: Candidate Test Taking Flow
    // ----------------------------------------------------
    console.log(`\n${colors.bold}4. Candidate Flow & MCQ Assessment${colors.reset}`);

    // Candidate views test overview by link
    const candidateInfoRes = await req(`/api/candidate/test-info/${shareableLink}`);
    assert(candidateInfoRes.status === 200, "GET /api/candidate/test-info/:link returns test overview");
    assert(candidateInfoRes.body.data.questionsCount === 3, "Returns question count without questions");
    assert(candidateInfoRes.body.data.questions === undefined, "Zero questions or answers leaked");

    // Candidate starts test attempt
    const startAttemptRes = await req("/api/candidate/start-attempt", {
      method: "POST",
      body: JSON.stringify({
        shareableLink,
        name: "Alice Candidate",
        email: `alice_${timestamp}@student.edu`,
        phone: "+1234567890",
      }),
    });
    assert(startAttemptRes.status === 200, "POST /api/candidate/start-attempt returns 200");
    assert(Boolean(startAttemptRes.body.data.attemptId), "Returns unique attemptId");
    assert(Boolean(startAttemptRes.body.data.sessionId), "Returns unique proctoring sessionId");
    assert(startAttemptRes.body.data.questions.length === 3, "Returns 3 questions for candidate");
    assert(
      startAttemptRes.body.data.questions[0].correctAnswer === undefined,
      "CRITICAL: correctAnswer is stripped from candidate questions"
    );

    candidateAttemptId = startAttemptRes.body.data.attemptId;
    candidateSessionId = startAttemptRes.body.data.sessionId;

    const q1 = startAttemptRes.body.data.questions[0];
    const q2 = startAttemptRes.body.data.questions[1];
    const q3 = startAttemptRes.body.data.questions[2];

    // Candidate answers Question 1 (Correct: Option 1)
    const ans1Res = await req("/api/candidate/answer", {
      method: "POST",
      body: JSON.stringify({
        attemptId: candidateAttemptId,
        questionId: q1._id,
        selectedOption: 1, // Correct (2 marks)
      }),
    });
    assert(ans1Res.status === 200, "POST /api/candidate/answer saves Question 1 answer");

    // Candidate answers Question 2 (Incorrect: Option 0)
    const ans2Res = await req("/api/candidate/answer", {
      method: "POST",
      body: JSON.stringify({
        attemptId: candidateAttemptId,
        questionId: q2._id,
        selectedOption: 0, // Incorrect (0 marks)
      }),
    });
    assert(ans2Res.status === 200, "POST /api/candidate/answer saves Question 2 answer");

    // Candidate answers Question 3 (Correct: Option 0)
    const ans3Res = await req("/api/candidate/answer", {
      method: "POST",
      body: JSON.stringify({
        attemptId: candidateAttemptId,
        questionId: q3._id,
        selectedOption: 0, // Correct (2 marks)
      }),
    });
    assert(ans3Res.status === 200, "POST /api/candidate/answer saves Question 3 answer");

    // Check candidate attempt status
    const statusRes = await req(`/api/candidate/attempt/${candidateAttemptId}`);
    assert(statusRes.status === 200, "GET /api/candidate/attempt/:attemptId returns status");
    assert(statusRes.body.data.answeredQuestionsCount === 3, "Attempt status reports 3 questions answered");
    assert(statusRes.body.data.remainingSeconds > 0, "Timer shows remaining countdown seconds");

    // Candidate submits test
    const submitRes = await req("/api/candidate/submit", {
      method: "POST",
      body: JSON.stringify({
        attemptId: candidateAttemptId,
      }),
    });
    assert(submitRes.status === 200, "POST /api/candidate/submit successfully computes score");
    assert(submitRes.body.data.score === 4, "Score accurately calculated: 4 / 5 marks (Q1: 2 + Q3: 2)");
    assert(submitRes.body.data.percentage === 80, "Percentage accurately computed (80%)");
    assert(submitRes.body.data.status === "submitted", "Attempt status updated to 'submitted'");

    // ----------------------------------------------------
    // TEST GROUP 5: Admin Dashboard & Proctoring Audit
    // ----------------------------------------------------
    console.log(`\n${colors.bold}5. Admin Dashboard & Audit Inspection${colors.reset}`);

    // Get attempts list for test
    const attemptsListRes = await req(`/api/dashboard/tests/${createdTestId}/attempts`, {
      headers: { Authorization: `Bearer ${educatorToken}` },
    });
    assert(attemptsListRes.status === 200, "GET /api/dashboard/tests/:testId/attempts returns list");
    assert(attemptsListRes.body.data.attempts.length >= 1, "Candidate attempt appears on dashboard");
    assert(attemptsListRes.body.data.attempts[0].score === 4, "Candidate score matches on dashboard");

    // Get attempt detailed breakdown (Questions vs Answers)
    const attemptDetailRes = await req(`/api/dashboard/attempts/${candidateAttemptId}`, {
      headers: { Authorization: `Bearer ${educatorToken}` },
    });
    assert(attemptDetailRes.status === 200, "GET /api/dashboard/attempts/:attemptId returns full breakdown");
    assert(attemptDetailRes.body.data.questions.length === 3, "Contains full 3 questions comparison");
    assert(attemptDetailRes.body.data.questions[0].isCorrect === true, "Q1 marked correct");
    assert(attemptDetailRes.body.data.questions[1].isCorrect === false, "Q2 marked incorrect");
    assert(attemptDetailRes.body.data.questions[2].isCorrect === true, "Q3 marked correct");

    // Get proctoring audit
    const procAuditRes = await req(`/api/dashboard/attempts/${candidateAttemptId}/proctoring`, {
      headers: { Authorization: `Bearer ${educatorToken}` },
    });
    assert(procAuditRes.status === 200, "GET /api/dashboard/attempts/:attemptId/proctoring returns audit log");
    assert(procAuditRes.body.data.session.status === "completed", "Session status is 'completed'");

    // ----------------------------------------------------
    // TEST GROUP 6: Real-Time Socket.IO & 3-Strike Rule
    // ----------------------------------------------------
    console.log(`\n${colors.bold}6. Socket.IO Real-Time Proctoring & 3-Strike Enforcement${colors.reset}`);

    // Start a second attempt specifically to test strike enforcement
    const attempt2Res = await req("/api/candidate/start-attempt", {
      method: "POST",
      body: JSON.stringify({
        shareableLink,
        name: "Bob Malpractice Tester",
        email: `bob_${timestamp}@test.org`,
      }),
    });
    secondAttemptId = attempt2Res.body.data.attemptId;
    secondSessionId = attempt2Res.body.data.sessionId;

    // Connect via WebSocket client
    const socket = ioClient(BASE_URL, {
      transports: ["websocket"],
      reconnection: false,
    });

    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("Socket connection timeout")), 4000);
      socket.on("connect", () => {
        clearTimeout(timeout);
        assert(socket.connected, "Socket.IO connected to backend");
        resolve();
      });
    });

    // Join proctoring room
    await new Promise((resolve) => {
      socket.emit("join_proctoring", {
        attemptId: secondAttemptId,
        sessionId: secondSessionId,
      });
      socket.on("joined_session", (data) => {
        assert(data.status === "connected", "Socket joined attempt room successfully");
        resolve();
      });
    });

    // Test Strike 1: emit client incident
    const strike1Promise = new Promise((resolve) => {
      socket.on("warning_issued", (data) => {
        if (data.strike === 1) {
          assert(data.strike === 1, "Strike 1 issued (warning_issued received)");
          assert(data.maxStrikes === 3, "Max strikes is 3");
          resolve();
        }
      });
    });

    socket.emit("client_incident", {
      attemptId: secondAttemptId,
      sessionId: secondSessionId,
      incidentType: "tab_switch",
      details: { tabUrl: "google.com" },
    });
    await strike1Promise;

    // Test Strike 2: simulate visual gaze deviation
    const strike2Promise = new Promise((resolve) => {
      socket.on("warning_issued", (data) => {
        if (data.strike === 2) {
          assert(data.strike === 2, "Strike 2 issued (warning_issued received)");
          resolve();
        }
      });
    });

    socket.emit("simulate_anomaly", {
      attemptId: secondAttemptId,
      sessionId: secondSessionId,
      anomalyType: "gaze_deviation",
      message: "Candidate looked away from screen",
    });
    await strike2Promise;

    // Test Strike 3: simulate secondary person -> MUST TRIGGER force_terminate!
    const strike3Promise = new Promise((resolve) => {
      socket.on("force_terminate", (data) => {
        assert(data.strikes === 3, "Strike 3 triggers force_terminate event");
        assert(Boolean(data.reason), `Termination reason: ${data.reason}`);
        resolve();
      });
    });

    socket.emit("simulate_anomaly", {
      attemptId: secondAttemptId,
      sessionId: secondSessionId,
      anomalyType: "multiple_faces",
      message: "Multiple faces detected in room",
    });
    await strike3Promise;

    // Verify session locked in DB
    const lockedStatusRes = await req(`/api/candidate/attempt/${secondAttemptId}`);
    assert(lockedStatusRes.body.data.status === "terminated", "Attempt status updated to 'terminated' in DB");
    assert(lockedStatusRes.body.data.strikes === 3, "Session strikes counter is 3/3");

    // Candidate tries to answer after termination -> MUST BE REJECTED
    const rejectAnswerRes = await req("/api/candidate/answer", {
      method: "POST",
      body: JSON.stringify({
        attemptId: secondAttemptId,
        questionId: q1._id,
        selectedOption: 0,
      }),
    });
    assert(rejectAnswerRes.status === 403, "POST /api/candidate/answer rejects terminated attempt (403 Forbidden)");

    // Proctoring audit shows all 3 anomalies with strikes
    const strikeAuditRes = await req(`/api/dashboard/attempts/${secondAttemptId}/proctoring`, {
      headers: { Authorization: `Bearer ${educatorToken}` },
    });
    assert(strikeAuditRes.body.data.session.status === "terminated", "Proctoring session status is 'terminated'");
    assert(strikeAuditRes.body.data.anomalySummary.strikesIssued === 3, "Proctoring audit reports 3 strikes issued");
    assert(strikeAuditRes.body.data.logs.length === 3, "Audit logs contain exactly 3 incident records");

    socket.disconnect();

  } catch (error) {
    console.error(`\n${colors.red}Test execution error:${colors.reset}`, error);
    failedCount++;
  } finally {
    server.close();
  }

  // ----------------------------------------------------
  // SUMMARY REPORT
  // ----------------------------------------------------
  console.log(`\n${colors.bold}${colors.cyan}====================================================${colors.reset}`);
  console.log(`${colors.bold}Test Suite Results:${colors.reset}`);
  console.log(`  Passed: ${colors.green}${passedCount}${colors.reset}`);
  console.log(`  Failed: ${failedCount === 0 ? colors.green + "0" : colors.red + failedCount}${colors.reset}`);
  console.log(`${colors.bold}${colors.cyan}====================================================\n${colors.reset}`);

  if (failedCount > 0) {
    process.exit(1);
  } else {
    process.exit(0);
  }
}

runTestSuite();
