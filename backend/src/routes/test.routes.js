import { Router } from "express";
import {
  createTest,
  getMyTests,
  getTestById,
  updateTest,
  deleteTest,
  toggleTestStatus,
  regenerateShareableLink,
} from "../controllers/test.controller.js";
import { verifyJWT, authorizeRoles } from "../middlewares/auth.middleware.js";

const router = Router();

// All test management routes require admin/educator authentication
router.use(verifyJWT, authorizeRoles("admin", "educator"));

router.route("/").post(createTest).get(getMyTests);
router.route("/:testId").get(getTestById).put(updateTest).delete(deleteTest);
router.patch("/:testId/toggle-status", toggleTestStatus);
router.post("/:testId/regenerate-link", regenerateShareableLink);

export default router;
