import mongoose from "mongoose";

const anomalyLogSchema = new mongoose.Schema(
  {
    attemptId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "TestAttempt",
      required: true,
      index: true,
    },
    sessionId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "ProctoringSession",
      required: true,
      index: true,
    },
    type: {
      type: String,
      enum: [
        "no_face_detected",
        "multiple_faces",
        "gaze_deviation",
        "head_pose_deviation",
        "secondary_voice",
        "speaking_detected",
        "audio_anomaly",
        "other",
      ],
      required: true,
    },
    severity: {
      type: String,
      enum: ["low", "medium", "high"],
      default: "medium",
    },
    message: {
      type: String,
      required: true,
    },
    confidence: {
      type: Number,
      min: 0,
      max: 1,
      default: null,
    },
    strikeIssued: {
      type: Boolean,
      default: false,
    },
    strikeNumber: {
      type: Number,
      default: null,
      min: 1,
      max: 3,
    },
    evidenceUrl: {
      type: String,
      default: null,
    },
    detectedAt: {
      type: Date,
      default: Date.now,
      index: true,
    },
    metadata: {
      type: mongoose.Schema.Types.Mixed,
      default: {},
    },
  },
  { timestamps: true }
);

export default mongoose.model("AnomalyLog", anomalyLogSchema);
