// docs/src/auth/better-auth.js

import { betterAuth } from "better-auth";
import { postgresAdapter } from "@better-auth/postgres-adapter";
import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";

// For now, using a mock database setup
// In production, you would connect to your actual database
const pool = new Pool({
  connectionString: process.env.DATABASE_URL || "postgresql://localhost:5432/humanoid_book",
});

const db = drizzle(pool);

export const auth = betterAuth({
  database: postgresAdapter(db, {
    provider: "pg",
  }),
  secret: process.env.BETTER_AUTH_SECRET || "your-secret-key-change-in-production",
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false,
  },
  socialProviders: {
    // Add social providers if needed
  },
  // Add custom fields for user background information
  user: {
    additionalFields: {
      softwareExperience: {
        type: "string",
        required: false,
      },
      hardwareExperience: {
        type: "string",
        required: false,
      },
      programmingLanguages: {
        type: "string",
        required: false,
      },
      roboticsExperience: {
        type: "string",
        required: false,
      },
      primaryGoal: {
        type: "string",
        required: false,
      },
      availableTime: {
        type: "string",
        required: false,
      },
      preferredLearningStyle: {
        type: "string",
        required: false,
      },
    },
  },
});