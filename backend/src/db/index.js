import mongoose from "mongoose";

const connectDB = async () => {
  try {
    const mongoUri =
      process.env.MONGO_URI ||
      process.env.MONGODB_URI ||
      "mongodb://127.0.0.1:27017/intelliscreen";

    const connectionInstance = await mongoose.connect(mongoUri, {
      dbName: process.env.DB_NAME || "intelliscreen",
    });

    console.log(`\n MongoDB Connected! DB HOST: ${connectionInstance.connection.host}`);
    return connectionInstance;
  } catch (error) {
    console.error("MongoDB connection failed:", error.message);
    process.exit(1);
  }
};

export default connectDB;
