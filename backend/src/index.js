import http from "http";
import dotenv from "dotenv";
import connectDB from "./db/index.js";
import app from "./app.js";
import { initializeSocket } from "./socket/proctoring.socket.js";

dotenv.config({
  path: "./.env",
});

const PORT = process.env.PORT || 5000;

// Create standard HTTP server wrapping Express app
const server = http.createServer(app);

// Initialize real-time Socket.IO proctoring engine
initializeSocket(server);

connectDB()
  .then(() => {
    server.listen(PORT, () => {
      console.log(`🚀 IntelliScreen Server is running on port: ${PORT}`);
    });
  })
  .catch((err) => {
    console.error("Failed to connect to the database:", err.message);
  });
