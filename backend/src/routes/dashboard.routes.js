import { Router } from "express";
import {
  getTestAttempts,
  getAttemptDetails,
  getProctoringAudit,
} from "../controllers/dashboard.controller.js";
import { verifyJWT, authorizeRoles } from "../middlewares/auth.middleware.js";

const router = Router();

// Dashboard routes require authenticated admin/educator
router.use(verifyJWT, authorizeRoles("admin", "educator"));

router.get("/tests/:testId/attempts", getTestAttempts);
router.get("/attempts/:attemptId", getAttemptDetails);
router.get("/attempts/:attemptId/proctoring", getProctoringAudit);

export default router;
