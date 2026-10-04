import mongoose from "mongoose";

const answerSchema = new mongoose.Schema(
    {
        questionId: {
            type: mongoose.Schema.Types.ObjectId,
            required: true,
        },
        selectedOption: {
            type: Number,
            default: null,
            min: 0,
        },
        isCorrect: {
            type: Boolean,
            default: null,
        },
        marksAwarded: {
            type: Number,
            default: 0,
        },
        answeredAt: Date,
    },
    { _id: false }
);

const testAttemptSchema = new mongoose.Schema(
    {
        testId: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "Test",
            required: true,
            index: true,
        },
        candidateId: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "Candidate",
            required: true,
            index: true,
        },
        answers: [answerSchema],
        startedAt: {
            type: Date,
            default: Date.now,
        },
        submittedAt: Date,
        expiresAt: {
            type: Date,
            required: true,
        },
        score: {
            type: Number,
            default: 0,
            min: 0,
        },
        totalMarks: {
            type: Number,
            required: true,
            min: 0,
        },
        status: {
            type: String,
            enum: [
                "in_progress",
                "submitted",
                "terminated",
                "expired",
            ],
            default: "in_progress",
            index: true,
        },
    },
    { timestamps: true }
);

export default mongoose.model("TestAttempt", testAttemptSchema);