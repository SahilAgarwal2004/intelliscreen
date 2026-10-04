import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import authRoutes from "./routes/auth.routes.js";
import testRoutes from "./routes/test.routes.js";
import candidateRoutes from "./routes/candidate.routes.js";
import dashboardRoutes from "./routes/dashboard.routes.js";
import { errorHandler } from "./middlewares/error.middleware.js";

const app = express();

// Global Middlewares
app.use(
  cors({
    origin: process.env.CORS_ORIGIN || "*",
    credentials: true,
  })
);

app.use(express.json({ limit: "16mb" }));
app.use(express.urlencoded({ extended: true, limit: "16mb" }));
app.use(cookieParser());

// Health & System Info
app.get("/health", (req, res) => {
  res.status(200).json({
    status: "ok",
    service: "IntelliScreen Backend API",
    timestamp: new Date().toISOString(),
  });
});

// Mount Application Routes
app.use("/api/auth", authRoutes);
app.use("/api/tests", testRoutes);
app.use("/api/candidate", candidateRoutes);
app.use("/api/dashboard", dashboardRoutes);

// Centralized Error Handler
app.use(errorHandler);

export default app;
