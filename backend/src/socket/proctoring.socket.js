import axios from "axios";
import { Server } from "socket.io";
import AnomalyLog from "../models/AnomalyLog.js";
import ProctoringSession from "../models/ProctoringSession.js";
import TestAttempt from "../models/TestAttempt.js";

// Cooldown between successive strikes to allow candidate to see warning and correct posture
const STRIKE_COOLDOWN_MS = process.env.STRIKE_COOLDOWN_MS
  ? parseInt(process.env.STRIKE_COOLDOWN_MS, 10)
  : 15000;

// Helper to record an anomaly and enforce the 3-strike rule
export const handleAnomalyEnforcement = async (
  io,
  socket,
  { attemptId, sessionId, anomalyType, message, severity = "medium", confidence = 0.9, metadata = {} }
) => {
  try {
    const session = await ProctoringSession.findOne({
      $or: [{ _id: sessionId }, { attemptId }],
    });

    if (!session) {
      console.warn(`[Proctoring] Session not found for attempt: ${attemptId}`);
      return;
    }

    if (session.status === "terminated" || session.status === "completed") {
      return;
    }

    // Fairness cooldown check: prevent rapid-fire strikes (e.g. within 15 seconds)
    if (!metadata.bypassCooldown && session.lastStrikeAt) {
      const elapsedMs = Date.now() - new Date(session.lastStrikeAt).getTime();
      if (elapsedMs < STRIKE_COOLDOWN_MS) {
        console.log(
          `[Proctoring] Strike cooldown active (${Math.round((STRIKE_COOLDOWN_MS - elapsedMs) / 1000)}s remaining). Anomaly logged without incrementing strikes.`
        );

        const suppressedLog = await AnomalyLog.create({
          attemptId: session.attemptId,
          sessionId: session._id,
          type: anomalyType,
          severity,
          message: `[Active Warning] ${message}`,
          confidence,
          strikeIssued: false,
          strikeNumber: session.strikes,
          detectedAt: new Date(),
          metadata: { ...metadata, cooldownSuppressed: true, elapsedMs },
        });

        return { session, anomalyLog: suppressedLog, cooldownSuppressed: true };
      }
    }

    // Increment strikes
    const newStrikeCount = session.strikes + 1;
    session.strikes = Math.min(newStrikeCount, 3);
    session.lastStrikeAt = new Date();
    session.lastActivityAt = new Date();

    const isTerminated = session.strikes >= 3;

    if (isTerminated) {
      session.status = "terminated";
      session.terminatedAt = new Date();
      session.terminationReason = `Exceeded maximum strikes (3/3): ${message}`;

      // Update attempt status
      await TestAttempt.findByIdAndUpdate(session.attemptId, {
        status: "terminated",
      });
    } else {
      session.status = "warning";
    }

    await session.save();

    // Log the anomaly in database
    const anomalyLog = await AnomalyLog.create({
      attemptId: session.attemptId,
      sessionId: session._id,
      type: anomalyType,
      severity,
      message,
      confidence,
      strikeIssued: true,
      strikeNumber: session.strikes,
      detectedAt: new Date(),
      metadata,
    });

    const roomName = `attempt_${session.attemptId}`;

    if (isTerminated) {
      console.log(`[Proctoring] Strike 3/3 reached. Force terminating attempt ${session.attemptId}`);
      io.to(roomName).emit("force_terminate", {
        reason: session.terminationReason,
        strikes: session.strikes,
        maxStrikes: 3,
        anomaly: {
          type: anomalyType,
          message,
          detectedAt: anomalyLog.detectedAt,
        },
      });
    } else {
      console.log(`[Proctoring] Strike ${session.strikes}/3 issued for attempt ${session.attemptId}: ${anomalyType}`);
      io.to(roomName).emit("warning_issued", {
        strike: session.strikes,
        maxStrikes: 3,
        message,
        anomalyType,
        detectedAt: anomalyLog.detectedAt,
      });
    }

    return { session, anomalyLog };
  } catch (error) {
    console.error("[Proctoring] Error during anomaly enforcement:", error);
  }
};

export const initializeSocket = (httpServer) => {
  const io = new Server(httpServer, {
    cors: {
      origin: process.env.CORS_ORIGIN || "*",
      methods: ["GET", "POST"],
      credentials: true,
    },
    maxHttpBufferSize: 5e6, // 5MB for canvas frame snapshots
  });

  io.on("connection", (socket) => {
    console.log(`[Socket] Candidate client connected: ${socket.id}`);

    // Join room corresponding to candidate's test attempt
    socket.on("join_proctoring", async ({ attemptId, sessionId }) => {
      if (!attemptId) return;

      const roomName = `attempt_${attemptId}`;
      socket.join(roomName);
      socket.attemptId = attemptId;
      socket.sessionId = sessionId;

      try {
        await ProctoringSession.findOneAndUpdate(
          { attemptId },
          { socketId: socket.id, lastActivityAt: new Date() }
        );
        console.log(`[Socket] Socket ${socket.id} joined room ${roomName}`);
        socket.emit("joined_session", {
          attemptId,
          status: "connected",
          message: "Real-time proctoring connection established",
        });

        // Initialize session in Python AI service if configured
        const pythonServiceUrl = process.env.PYTHON_AI_SERVICE_URL;
        if (pythonServiceUrl) {
          axios
            .post(
              `${pythonServiceUrl}/sessions/start`,
              {
                session_id: String(sessionId || attemptId),
                candidate_id: `cand_${attemptId}`,
                test_id: "mcq_test",
              },
              { timeout: 2000 }
            )
            .catch(() => {});
        }
      } catch (err) {
        console.error("[Socket] Error registering socketId in session:", err.message);
      }
    });

    // Ingest periodic webcam frame snapshots (1-2 FPS Base64 string)
    socket.on("media_frame", async (data) => {
      const { attemptId, sessionId, frameBase64, timestamp = Date.now(), frameIndex = 0 } = data || {};

      if (!attemptId || !frameBase64) return;

      const pythonServiceUrl = process.env.PYTHON_AI_SERVICE_URL;

      // If Python AI microservice is configured, forward frame
      if (pythonServiceUrl) {
        try {
          const aiResponse = await axios.post(
            `${pythonServiceUrl}/sessions/${sessionId || attemptId}/frame-base64`,
            {
              image_base64: frameBase64,
              frame_index: frameIndex,
              timestamp_ms: timestamp,
            },
            { timeout: 2500 }
          );

          const {
            face_detected,
            face_count,
            is_looking_away,
            secondary_person_detected,
            phone_detected,
            is_malpractice_flagged,
            severity,
          } = aiResponse.data;

          if (is_malpractice_flagged) {
            let anomalyType = "other";
            let message = "Suspicious behavior detected";

            if (phone_detected) {
              anomalyType = "other";
              message = "Mobile phone or unauthorized electronic device detected in frame";
            } else if (!face_detected) {
              anomalyType = "no_face_detected";
              message = "Candidate face is not detected in camera frame";
            } else if (secondary_person_detected || face_count > 1) {
              anomalyType = "multiple_faces";
              message = "Multiple faces / unauthorized person detected in frame";
            } else if (is_looking_away) {
              anomalyType = "gaze_deviation";
              message = "Candidate is looking away from the screen for prolonged periods";
            }

            await handleAnomalyEnforcement(io, socket, {
              attemptId,
              sessionId,
              anomalyType,
              message,
              severity: severity || "medium",
            });
          }
        } catch (axiosErr) {
          // If Python AI microservice is still under work or offline, do not crash Node.js
          // Just silently catch or log debug info
        }
      }
    });

    // Ingest browser events (Tab switch, Fullscreen exit, Blur)
    socket.on("client_incident", async ({ attemptId, sessionId, incidentType, details }) => {
      if (!attemptId) return;

      let anomalyType = "other";
      let message = "Candidate triggered client-side exam irregularity";

      if (incidentType === "tab_switch") {
        anomalyType = "other";
        message = "Candidate switched browser tabs or minimized test window";
      } else if (incidentType === "fullscreen_exit") {
        anomalyType = "other";
        message = "Candidate exited full-screen mode";
      }

      await handleAnomalyEnforcement(io, socket, {
        attemptId,
        sessionId,
        anomalyType,
        message,
        severity: "medium",
        metadata: { incidentType, ...(details || {}) },
      });
    });

    // Developer / manual testing trigger: simulate an anomaly to test the 3-strike rule
    socket.on("simulate_anomaly", async ({ attemptId, sessionId, anomalyType, message }) => {
      if (!attemptId) return;
      await handleAnomalyEnforcement(io, socket, {
        attemptId,
        sessionId,
        anomalyType: anomalyType || "no_face_detected",
        message: message || "Simulated test violation",
        metadata: { bypassCooldown: true },
      });
    });

    // Handle client disconnect
    socket.on("disconnect", async () => {
      console.log(`[Socket] Client disconnected: ${socket.id}`);
      if (socket.attemptId) {
        try {
          const session = await ProctoringSession.findOne({
            attemptId: socket.attemptId,
          });
          if (session && session.status !== "terminated" && session.status !== "completed") {
            session.status = "disconnected";
            session.lastActivityAt = new Date();
            await session.save();
          }
        } catch (err) {
          console.error("[Socket] Disconnect state update error:", err.message);
        }
      }
    });
  });

  return io;
};
