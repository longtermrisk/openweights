-- Add 'custom' to the job_type enum
ALTER TYPE "public"."job_type" ADD VALUE IF NOT EXISTS 'custom';
