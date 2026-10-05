import crypto from "crypto";
import Test from "../models/Test.js";
import TestAttempt from "../models/TestAttempt.js";
import { ApiError } from "../utils/ApiError.js";
import { ApiResponse } from "../utils/ApiResponse.js";
import { asyncHandler } from "../utils/asyncHandler.js";

// Helper to generate a clean, collision-resistant shareable code
const generateShareableCode = () => {
  return crypto.randomBytes(6).toString("hex");
};

// @desc    Create a new MCQ test with questions
// @route   POST /api/tests
// @access  Private (Admin / Educator)
export const createTest = asyncHandler(async (req, res) => {
  const {
    title,
    description,
    questions,
    duration,
    maxAttempts = 1,
    startTime,
    endTime,
  } = req.body || {};

  if (!title || !duration || !questions || !Array.isArray(questions)) {
    throw new ApiError(
      400,
      "Title, duration (in minutes), and questions array are required"
    );
  }

  if (questions.length === 0) {
    throw new ApiError(400, "A test must contain at least one question");
  }

  // Validate each question
  for (let i = 0; i < questions.length; i++) {
    const q = questions[i];
    if (!q.questionText || !q.options || !Array.isArray(q.options)) {
      throw new ApiError(
        400,
        `Question ${i + 1} must have questionText and options array`
      );
    }
    if (q.options.length < 2) {
      throw new ApiError(
        400,
        `Question ${i + 1} must have at least 2 options`
      );
    }
    if (
      typeof q.correctAnswer !== "number" ||
      q.correctAnswer < 0 ||
      q.correctAnswer >= q.options.length
    ) {
      throw new ApiError(
        400,
        `Question ${i + 1} correctAnswer must be a valid option index (0 to ${
          q.options.length - 1
        })`
      );
    }
  }

  let shareableLink = generateShareableCode();
  // Ensure uniqueness
  while (await Test.findOne({ shareableLink })) {
    shareableLink = generateShareableCode();
  }

  const test = await Test.create({
    title: title.trim(),
    description: description ? description.trim() : "",
    createdBy: req.user._id,
    questions,
    duration: Number(duration),
    maxAttempts: Number(maxAttempts) || 1,
    shareableLink,
    isActive: true,
    startTime: startTime ? new Date(startTime) : undefined,
    endTime: endTime ? new Date(endTime) : undefined,
  });

  return res
    .status(201)
    .json(new ApiResponse(201, test, "Test created successfully"));
});

// @desc    Get all tests created by logged-in admin
// @route   GET /api/tests
// @access  Private (Admin / Educator)
export const getMyTests = asyncHandler(async (req, res) => {
  const tests = await Test.find({ createdBy: req.user._id })
    .sort({ createdAt: -1 })
    .lean();

  // Attach attempt count for each test
  const testsWithStats = await Promise.all(
    tests.map(async (test) => {
      const attemptsCount = await TestAttempt.countDocuments({
        testId: test._id,
      });
      const totalMarks = test.questions.reduce(
        (sum, q) => sum + (q.marks || 1),
        0
      );

      return {
        ...test,
        totalQuestions: test.questions.length,
        totalMarks,
        attemptsCount,
      };
    })
  );

  return res
    .status(200)
    .json(
      new ApiResponse(
        200,
        testsWithStats,
        "Fetched admin tests successfully"
      )
    );
});

// @desc    Get single test details by ID (Full details for creator)
// @route   GET /api/tests/:testId
// @access  Private (Admin / Educator)
export const getTestById = asyncHandler(async (req, res) => {
  const { testId } = req.params;

  const test = await Test.findOne({
    _id: testId,
    createdBy: req.user._id,
  });

  if (!test) {
    throw new ApiError(404, "Test not found or access denied");
  }

  const totalMarks = test.questions.reduce(
    (sum, q) => sum + (q.marks || 1),
    0
  );

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        ...test.toObject(),
        totalMarks,
      },
      "Test retrieved successfully"
    )
  );
});

// @desc    Update test details
// @route   PUT /api/tests/:testId
// @access  Private (Admin / Educator)
export const updateTest = asyncHandler(async (req, res) => {
  const { testId } = req.params;
  const {
    title,
    description,
    questions,
    duration,
    isActive,
    maxAttempts,
    startTime,
    endTime,
  } = req.body;

  const test = await Test.findOne({
    _id: testId,
    createdBy: req.user._id,
  });

  if (!test) {
    throw new ApiError(404, "Test not found or access denied");
  }

  if (title) test.title = title.trim();
  if (description !== undefined) test.description = description.trim();
  if (duration) test.duration = Number(duration);
  if (isActive !== undefined) test.isActive = Boolean(isActive);
  if (maxAttempts) test.maxAttempts = Number(maxAttempts);
  if (startTime) test.startTime = new Date(startTime);
  if (endTime) test.endTime = new Date(endTime);

  if (questions && Array.isArray(questions)) {
    if (questions.length === 0) {
      throw new ApiError(400, "A test must contain at least one question");
    }
    for (let i = 0; i < questions.length; i++) {
      const q = questions[i];
      if (!q.questionText || !q.options || q.options.length < 2) {
        throw new ApiError(
          400,
          `Question ${i + 1} must have questionText and at least 2 options`
        );
      }
      if (
        typeof q.correctAnswer !== "number" ||
        q.correctAnswer < 0 ||
        q.correctAnswer >= q.options.length
      ) {
        throw new ApiError(
          400,
          `Question ${i + 1} correctAnswer is out of options bounds`
        );
      }
    }
    test.questions = questions;
  }

  await test.save();

  return res
    .status(200)
    .json(new ApiResponse(200, test, "Test updated successfully"));
});

// @desc    Toggle test active status
// @route   PATCH /api/tests/:testId/toggle-status
// @access  Private (Admin / Educator)
export const toggleTestStatus = asyncHandler(async (req, res) => {
  const { testId } = req.params;

  const test = await Test.findOne({
    _id: testId,
    createdBy: req.user._id,
  });

  if (!test) {
    throw new ApiError(404, "Test not found or access denied");
  }

  test.isActive = !test.isActive;
  await test.save();

  return res.status(200).json(
    new ApiResponse(
      200,
      { testId: test._id, isActive: test.isActive },
      `Test is now ${test.isActive ? "active" : "inactive"}`
    )
  );
});

// @desc    Delete a test
// @route   DELETE /api/tests/:testId
// @access  Private (Admin / Educator)
export const deleteTest = asyncHandler(async (req, res) => {
  const { testId } = req.params;

  const test = await Test.findOneAndDelete({
    _id: testId,
    createdBy: req.user._id,
  });

  if (!test) {
    throw new ApiError(404, "Test not found or access denied");
  }

  return res
    .status(200)
    .json(new ApiResponse(200, null, "Test deleted successfully"));
});

// @desc    Regenerate a fresh unique shareable link for a test
// @route   POST /api/tests/:testId/regenerate-link
// @access  Private (Admin / Educator)
export const regenerateShareableLink = asyncHandler(async (req, res) => {
  const { testId } = req.params;

  const test = await Test.findOne({
    _id: testId,
    createdBy: req.user._id,
  });

  if (!test) {
    throw new ApiError(404, "Test not found or access denied");
  }

  let newShareableLink = generateShareableCode();
  while (await Test.findOne({ shareableLink: newShareableLink })) {
    newShareableLink = generateShareableCode();
  }

  test.shareableLink = newShareableLink;
  await test.save();

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        testId: test._id,
        shareableLink: test.shareableLink,
      },
      "Unique test link regenerated successfully"
    )
  );
});
