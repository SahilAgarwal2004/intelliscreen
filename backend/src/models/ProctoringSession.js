import mongoose from "mongoose";

const proctoringSessionSchema = new mongoose.Schema(
    {
        attemptId: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "TestAttempt",
            required: true,
            unique: true,
            index: true,
        },
        socketId: {
            type: String,
            default: null,
        },
        status: {
            type: String,
            enum: [
                "active",
                "warning",
                "terminated",
                "completed",
                "disconnected",
            ],
            default: "active",
            index: true,
        },
        strikes: {
            type: Number,
            default: 0,
            min: 0,
            max: 3,
        },
        maxStrikes: {
            type: Number,
            default: 3,
        },
        lastActivityAt: {
            type: Date,
            default: Date.now,
        },
        terminatedAt: {
            type: Date,
            default: null,
        },
        terminationReason: {
            type: String,
            default: null,
        },
    },
    { timestamps: true }
);

export default mongoose.model(
    "ProctoringSession",
    proctoringSessionSchema
);