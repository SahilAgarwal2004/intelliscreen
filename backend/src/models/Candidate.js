import mongoose from "mongoose";

const candidateSchema = new mongoose.Schema(
    {
        name: {
            type: String,
            required: true,
            trim: true,
        },
        email: {
            type: String,
            lowercase: true,
            trim: true,
            default: null,
        },
        phone: {
            type: String,
            default: null,
        },
    },
    { timestamps: true }
);

export default mongoose.model("Candidate", candidateSchema);