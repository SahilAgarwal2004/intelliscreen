import Candidate from "../models/Candidate.js";
import ProctoringSession from "../models/ProctoringSession.js";
import Test from "../models/Test.js";
import TestAttempt from "../models/TestAttempt.js";
import { ApiError } from "../utils/ApiError.js";
import { ApiResponse } from "../utils/ApiResponse.js";
import { asyncHandler } from "../utils/asyncHandler.js";

// @desc    Get public test overview by shareable link
// @route   GET /api/candidate/test-info/:shareableLink
// @access  Public
export const getTestInfoByLink = asyncHandler(async (req, res) => {
  const { shareableLink } = req.params;

  let cleanCode = (shareableLink || "").trim().split("?")[0].split("#")[0];
  if (cleanCode.includes("/")) {
    const parts = cleanCode.split("/").filter(Boolean);
    cleanCode = parts[parts.length - 1];
  }

  const query = {
    $or: [
      { shareableLink: cleanCode },
      ...(cleanCode.match(/^[0-9a-fA-F]{24}$/) ? [{ _id: cleanCode }] : []),
    ],
  };

  const test = await Test.findOne(query);

  if (!test) {
    throw new ApiError(404, "Test not found. Please check your invitation link.");
  }

  if (!test.isActive) {
    throw new ApiError(403, "This test is currently inactive or closed by the administrator.");
  }

  const now = new Date();
  if (test.startTime && now < test.startTime) {
    throw new ApiError(403, `This test will be available starting ${test.startTime.toISOString()}`);
  }
  if (test.endTime && now > test.endTime) {
    throw new ApiError(403, "This test submission window has already ended.");
  }

  const totalMarks = test.questions.reduce((sum, q) => sum + (q.marks || 1), 0);

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        testId: test._id,
        title: test.title,
        description: test.description,
        durationMinutes: test.duration,
        questionsCount: test.questions.length,
        totalMarks,
        maxAttempts: test.maxAttempts,
        startTime: test.startTime,
        endTime: test.endTime,
      },
      "Test details fetched successfully"
    )
  );
});

// @desc    Register candidate details & start proctored test attempt
// @route   POST /api/candidate/start-attempt
// @access  Public
export const startTestAttempt = asyncHandler(async (req, res) => {
  const { shareableLink, name, email, phone } = req.body;

  if (!shareableLink || !name || !email) {
    throw new ApiError(400, "Shareable link, candidate name, and email are required");
  }

  let cleanCode = (shareableLink || "").trim().split("?")[0].split("#")[0];
  if (cleanCode.includes("/")) {
    const parts = cleanCode.split("/").filter(Boolean);
    cleanCode = parts[parts.length - 1];
  }

  const query = {
    $or: [
      { shareableLink: cleanCode },
      ...(cleanCode.match(/^[0-9a-fA-F]{24}$/) ? [{ _id: cleanCode }] : []),
    ],
  };

  const test = await Test.findOne(query);
  if (!test) {
    throw new ApiError(404, "Test not found");
  }

  if (!test.isActive) {
    throw new ApiError(403, "This test is currently inactive");
  }

  const now = new Date();
  if (test.startTime && now < test.startTime) {
    throw new ApiError(403, "This test has not started yet");
  }
  if (test.endTime && now > test.endTime) {
    throw new ApiError(403, "This test has already expired");
  }

  // Find or create Candidate record
  let candidate = await Candidate.findOne({
    email: email.toLowerCase().trim(),
  });

  if (!candidate) {
    candidate = await Candidate.create({
      name: name.trim(),
      email: email.toLowerCase().trim(),
      phone: phone ? phone.trim() : null,
    });
  } else {
    // Update name or phone if provided
    candidate.name = name.trim();
    if (phone) candidate.phone = phone.trim();
    await candidate.save();
  }

  // Check attempt limits
  const completedAttemptsCount = await TestAttempt.countDocuments({
    testId: test._id,
    candidateId: candidate._id,
    status: { $in: ["submitted", "terminated", "expired"] },
  });

  if (completedAttemptsCount >= test.maxAttempts) {
    throw new ApiError(
      403,
      `You have reached the maximum allowed attempts (${test.maxAttempts}) for this test`
    );
  }

  // Check if candidate has an existing active in-progress attempt
  let activeAttempt = await TestAttempt.findOne({
    testId: test._id,
    candidateId: candidate._id,
    status: "in_progress",
  });

  let proctoringSession;

  if (activeAttempt) {
    // Check if active attempt is already expired
    if (now > activeAttempt.expiresAt) {
      activeAttempt.status = "expired";
      await activeAttempt.save();
      throw new ApiError(403, "Your previous attempt has expired");
    }

    proctoringSession = await ProctoringSession.findOne({
      attemptId: activeAttempt._id,
    });

    if (proctoringSession && proctoringSession.status === "terminated") {
      activeAttempt.status = "terminated";
      await activeAttempt.save();
      throw new ApiError(
        403,
        "Your attempt was terminated due to proctoring policy violations"
      );
    }
  } else {
    const totalMarks = test.questions.reduce(
      (sum, q) => sum + (q.marks || 1),
      0
    );

    const expiresAt = new Date(Date.now() + test.duration * 60 * 1000);

    activeAttempt = await TestAttempt.create({
      testId: test._id,
      candidateId: candidate._id,
      answers: [],
      startedAt: now,
      expiresAt,
      score: 0,
      totalMarks,
      status: "in_progress",
    });

    proctoringSession = await ProctoringSession.create({
      attemptId: activeAttempt._id,
      status: "active",
      strikes: 0,
      maxStrikes: 3,
      lastActivityAt: now,
    });
  }

  // Sanitize questions: strip `correctAnswer` before sending to candidate!
  const sanitizedQuestions = test.questions.map((q) => ({
    _id: q._id,
    questionText: q.questionText,
    options: q.options,
    marks: q.marks || 1,
  }));

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        attemptId: activeAttempt._id,
        sessionId: proctoringSession._id,
        candidate: {
          _id: candidate._id,
          name: candidate.name,
          email: candidate.email,
        },
        test: {
          testId: test._id,
          title: test.title,
          description: test.description,
          durationMinutes: test.duration,
          totalMarks: activeAttempt.totalMarks,
          startedAt: activeAttempt.startedAt,
          expiresAt: activeAttempt.expiresAt,
        },
        questions: sanitizedQuestions,
        answers: activeAttempt.answers,
        strikes: proctoringSession?.strikes || 0,
        maxStrikes: proctoringSession?.maxStrikes || 3,
      },
      "Test attempt started successfully"
    )
  );
});

// @desc    Record/Update answer for a specific question
// @route   POST /api/candidate/answer
// @access  Public
export const saveAnswer = asyncHandler(async (req, res) => {
  const { attemptId, questionId, selectedOption } = req.body;

  if (!attemptId || !questionId || selectedOption === undefined) {
    throw new ApiError(400, "attemptId, questionId, and selectedOption are required");
  }

  const attempt = await TestAttempt.findById(attemptId);
  if (!attempt) {
    throw new ApiError(404, "Test attempt not found");
  }

  if (attempt.status !== "in_progress") {
    throw new ApiError(
      403,
      `Cannot answer questions. Attempt is currently '${attempt.status}'`
    );
  }

  // Check expiration
  if (new Date() > attempt.expiresAt) {
    attempt.status = "expired";
    await attempt.save();
    throw new ApiError(403, "Time limit has expired for this test attempt");
  }

  // Check proctoring session status
  const session = await ProctoringSession.findOne({ attemptId });
  if (session && session.status === "terminated") {
    attempt.status = "terminated";
    await attempt.save();
    throw new ApiError(
      403,
      "Test attempt is terminated due to proctoring strikes limit"
    );
  }

  const test = await Test.findById(attempt.testId);
  if (!test) {
    throw new ApiError(404, "Test reference not found");
  }

  const question = test.questions.id(questionId);
  if (!question) {
    throw new ApiError(404, "Question not found in test");
  }

  if (selectedOption < 0 || selectedOption >= question.options.length) {
    throw new ApiError(400, "selectedOption index is out of bounds");
  }

  const isCorrect = selectedOption === question.correctAnswer;
  const marksAwarded = isCorrect ? (question.marks || 1) : 0;

  // Check if question already answered
  const existingAnswerIndex = attempt.answers.findIndex(
    (a) => a.questionId.toString() === questionId.toString()
  );

  const answerPayload = {
    questionId,
    selectedOption,
    isCorrect,
    marksAwarded,
    answeredAt: new Date(),
  };

  if (existingAnswerIndex > -1) {
    attempt.answers[existingAnswerIndex] = answerPayload;
  } else {
    attempt.answers.push(answerPayload);
  }

  await attempt.save();

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        questionId,
        selectedOption,
        totalAnswered: attempt.answers.length,
      },
      "Answer saved successfully"
    )
  );
});

// @desc    Final test submission & score calculation
// @route   POST /api/candidate/submit
// @access  Public
export const submitTest = asyncHandler(async (req, res) => {
  const { attemptId } = req.body;

  if (!attemptId) {
    throw new ApiError(400, "attemptId is required");
  }

  const attempt = await TestAttempt.findById(attemptId);
  if (!attempt) {
    throw new ApiError(404, "Test attempt not found");
  }

  if (attempt.status === "submitted") {
    return res.status(200).json(
      new ApiResponse(
        200,
        {
          attemptId: attempt._id,
          score: attempt.score,
          totalMarks: attempt.totalMarks,
          status: attempt.status,
        },
        "Test was already submitted"
      )
    );
  }

  if (attempt.status === "terminated") {
    throw new ApiError(403, "This attempt was terminated due to proctoring strikes");
  }

  const test = await Test.findById(attempt.testId);
  if (!test) {
    throw new ApiError(404, "Test not found");
  }

  // Calculate final score
  let finalScore = 0;
  for (const ans of attempt.answers) {
    const question = test.questions.id(ans.questionId);
    if (question && ans.selectedOption === question.correctAnswer) {
      finalScore += question.marks || 1;
    }
  }

  attempt.score = finalScore;
  attempt.status = "submitted";
  attempt.submittedAt = new Date();
  await attempt.save();

  // Mark proctoring session as completed
  await ProctoringSession.findOneAndUpdate(
    { attemptId: attempt._id },
    { status: "completed", lastActivityAt: new Date() }
  );

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        attemptId: attempt._id,
        score: attempt.score,
        totalMarks: attempt.totalMarks,
        percentage:
          attempt.totalMarks > 0
            ? Math.round((attempt.score / attempt.totalMarks) * 100)
            : 0,
        status: attempt.status,
        submittedAt: attempt.submittedAt,
      },
      "Test submitted successfully"
    )
  );
});

// @desc    Get candidate's current attempt status and time remaining
// @route   GET /api/candidate/attempt/:attemptId
// @access  Public
export const getAttemptStatus = asyncHandler(async (req, res) => {
  const { attemptId } = req.params;

  const attempt = await TestAttempt.findById(attemptId).populate(
    "candidateId",
    "name email"
  );

  if (!attempt) {
    throw new ApiError(404, "Test attempt not found");
  }

  const session = await ProctoringSession.findOne({ attemptId: attempt._id });
  const now = new Date();

  // If time has run out and not marked yet
  if (attempt.status === "in_progress" && now > attempt.expiresAt) {
    attempt.status = "expired";
    await attempt.save();
    if (session && session.status !== "terminated") {
      session.status = "completed";
      await session.save();
    }
  }

  const remainingSeconds = Math.max(
    0,
    Math.floor((attempt.expiresAt - now) / 1000)
  );

  return res.status(200).json(
    new ApiResponse(
      200,
      {
        attemptId: attempt._id,
        status: attempt.status,
        remainingSeconds,
        startedAt: attempt.startedAt,
        expiresAt: attempt.expiresAt,
        answeredQuestionsCount: attempt.answers.length,
        strikes: session?.strikes || 0,
        maxStrikes: session?.maxStrikes || 3,
        sessionStatus: session?.status || "disconnected",
      },
      "Attempt status retrieved"
    )
  );
});
