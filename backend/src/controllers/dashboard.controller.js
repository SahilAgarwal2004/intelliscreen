import AnomalyLog from "../models/AnomalyLog.js";
import ProctoringSession from "../models/ProctoringSession.js";
import Test from "../models/Test.js";
import TestAttempt from "../models/TestAttempt.js";
import { ApiError } from "../utils/ApiError.js";
import { ApiResponse } from "../utils/ApiResponse.js";
import { asyncHandler } from "../utils/asyncHandler.js";

// @desc    Get all candidate attempts for a specific test
// @route   GET /api/dashboard/tests/:testId/attempts
// @access  Private (Admin / Educator)
export const getTestAttempts = asyncHandler(async (req, res) => {
  const { testId } = req.params;

  // Verify test ownership
  const test = await Test.findOne({
    _id: testId,
    createdBy: req.user._id,
  });

  if (!test) {
    throw new ApiError(404, "Test not found or access denied");
  }

  const attempts = await TestAttempt.find({ testId })
    .populate("candidateId", "name email phone")
    .sort({ startedAt: -1 })
    .lean();

  // Attach session strikes and anomaly counts
  const attemptsWithAudit = await Promise.all(
    attempts.map(async (att) => {
      const session = await ProctoringSession.findOne({
        attemptId: att._id,
      }).lean();

      const anomalyCount = await AnomalyLog.countDocuments({
        attemptId: att._id,
      });

      const percentage =
        att.totalMarks > 0
          ? Math.round((att.score / att.totalMarks) * 100)
          : 0;

      return {
        ...att,
        percentage,
        strikes: session?.strikes || 0,
        sessionStatus: session?.status || "unknown",
        anomalyCount,
      };
    })
  );

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        test: {
          _id: test._id,
          title: test.title,
          totalQuestions: test.questions.length,
          shareableLink: test.shareableLink,
        },
        attempts: attemptsWithAudit,
      },
      "Test attempts fetched successfully"
    )
  );
});

// @desc    Get full candidate attempt breakdown (questions vs answers)
// @route   GET /api/dashboard/attempts/:attemptId
// @access  Private (Admin / Educator)
export const getAttemptDetails = asyncHandler(async (req, res) => {
  const { attemptId } = req.params;

  const attempt = await TestAttempt.findById(attemptId)
    .populate("candidateId", "name email phone")
    .lean();

  if (!attempt) {
    throw new ApiError(404, "Test attempt not found");
  }

  const test = await Test.findOne({
    _id: attempt.testId,
    createdBy: req.user._id,
  }).lean();

  if (!test) {
    throw new ApiError(404, "Test not found or access denied");
  }

  // Correlate questions with candidate's answers
  const questionBreakdown = test.questions.map((q, idx) => {
    const candidateAnswer = attempt.answers.find(
      (a) => a.questionId.toString() === q._id.toString()
    );

    const isAnswered = Boolean(candidateAnswer);
    const selectedOption = isAnswered ? candidateAnswer.selectedOption : null;
    const isCorrect = isAnswered
      ? selectedOption === q.correctAnswer
      : false;
    const marksAwarded = isCorrect ? (q.marks || 1) : 0;

    return {
      questionNumber: idx + 1,
      questionId: q._id,
      questionText: q.questionText,
      options: q.options,
      correctAnswer: q.correctAnswer,
      correctOptionText: q.options[q.correctAnswer],
      selectedOption,
      selectedOptionText:
        selectedOption !== null && selectedOption !== undefined
          ? q.options[selectedOption]
          : null,
      isCorrect,
      marksAwarded,
      maxMarks: q.marks || 1,
      answeredAt: candidateAnswer?.answeredAt || null,
    };
  });

  const session = await ProctoringSession.findOne({
    attemptId: attempt._id,
  }).lean();

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        attempt: {
          _id: attempt._id,
          candidate: attempt.candidateId,
          score: attempt.score,
          totalMarks: attempt.totalMarks,
          percentage:
            attempt.totalMarks > 0
              ? Math.round((attempt.score / attempt.totalMarks) * 100)
              : 0,
          status: attempt.status,
          startedAt: attempt.startedAt,
          submittedAt: attempt.submittedAt,
          expiresAt: attempt.expiresAt,
        },
        test: {
          _id: test._id,
          title: test.title,
          duration: test.duration,
        },
        proctoring: {
          sessionId: session?._id,
          status: session?.status,
          strikes: session?.strikes || 0,
          maxStrikes: session?.maxStrikes || 3,
          terminationReason: session?.terminationReason,
        },
        questions: questionBreakdown,
      },
      "Attempt details retrieved successfully"
    )
  );
});

// @desc    Get proctoring session audit and anomaly timeline for an attempt
// @route   GET /api/dashboard/attempts/:attemptId/proctoring
// @access  Private (Admin / Educator)
export const getProctoringAudit = asyncHandler(async (req, res) => {
  const { attemptId } = req.params;

  const attempt = await TestAttempt.findById(attemptId).populate(
    "candidateId",
    "name email"
  );

  if (!attempt) {
    throw new ApiError(404, "Test attempt not found");
  }

  // Verify ownership of the test
  const test = await Test.findOne({
    _id: attempt.testId,
    createdBy: req.user._id,
  });

  if (!test) {
    throw new ApiError(404, "Access denied: Test does not belong to you");
  }

  const session = await ProctoringSession.findOne({ attemptId }).lean();
  const anomalyLogs = await AnomalyLog.find({ attemptId })
    .sort({ detectedAt: 1 })
    .lean();

  // Summary counts by anomaly type
  const anomalyStats = anomalyLogs.reduce((acc, log) => {
    acc[log.type] = (acc[log.type] || 0) + 1;
    return acc;
  }, {});

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        attempt: {
          _id: attempt._id,
          candidate: attempt.candidateId,
          status: attempt.status,
          score: attempt.score,
          totalMarks: attempt.totalMarks,
        },
        session: session || {
          status: "not_started",
          strikes: 0,
          maxStrikes: 3,
        },
        anomalySummary: {
          totalViolations: anomalyLogs.length,
          strikesIssued: session?.strikes || 0,
          byType: anomalyStats,
        },
        logs: anomalyLogs,
      },
      "Proctoring audit fetched successfully"
    )
  );
});
