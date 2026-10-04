import jwt from "jsonwebtoken";

export const generateAccessToken = (user) => {
  return jwt.sign(
    {
      _id: user._id,
      email: user.email,
      role: user.role,
      name: user.name,
    },
    process.env.JWT_SECRET || "intelliscreen_jwt_secret_key_default",
    {
      expiresIn: process.env.JWT_EXPIRES_IN || "7d",
    }
  );
};
