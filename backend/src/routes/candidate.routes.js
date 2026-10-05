import { Router } from "express";
import {
  getTestInfoByLink,
  startTestAttempt,
  saveAnswer,
  submitTest,
  getAttemptStatus,
} from "../controllers/candidate.controller.js";

const router = Router();

router.get("/test-info/:shareableLink", getTestInfoByLink);
router.post("/start-attempt", startTestAttempt);
router.post("/answer", saveAnswer);
router.post("/submit", submitTest);
router.get("/attempt/:attemptId", getAttemptStatus);

export default router;
