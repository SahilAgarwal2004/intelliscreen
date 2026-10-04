import mongoose from "mongoose";

const questionSchema = new mongoose.Schema({
    questionText: {
        type: String,
        required: true,
        trim: true,
    },
    options: {
        type: [String],
        required: true,
        validate: {
            validator: (options) => options.length >= 2,
            message: "At least two options are required",
        },
    },
    correctAnswer: {
        type: Number, // Index of the correct option
        required: true,
        min: 0,
    },
    marks: {
        type: Number,
        default: 1,
        min: 0,
    },
});

const testSchema = new mongoose.Schema(
    {
        title: {
            type: String,
            required: true,
            trim: true,
        },
        description: {
            type: String,
            default: "",
        },
        createdBy: {
            type: mongoose.Schema.Types.ObjectId,
            ref: "User",
            required: true,
        },
        questions: {
            type: [questionSchema],
            validate: {
                validator: (questions) => questions.length > 0,
                message: "A test must contain at least one question",
            },
        },
        duration: {
            type: Number, // Minutes
            required: true,
            min: 1,
        },
        shareableLink: {
            type: String,
            unique: true,
            required: true,
        },
        isActive: {
            type: Boolean,
            default: true,
        },
        startTime: Date,
        endTime: Date,
        maxAttempts: {
            type: Number,
            default: 1,
            min: 1,
        },
    },
    { timestamps: true }
);

export default mongoose.model("Test", testSchema);