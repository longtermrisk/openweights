-- Add cloud_type column to jobs table
-- Allows users to specify RunPod cloud provider type per job:
--   'SECURE'    = on-demand only (default, most reliable)
--   'ALL'       = on-demand + community cloud
--   'COMMUNITY' = community/spot cloud only (cheapest)

ALTER TABLE "public"."jobs"
    ADD COLUMN IF NOT EXISTS "cloud_type" text DEFAULT 'SECURE';

ALTER TABLE "public"."jobs"
    ADD CONSTRAINT IF NOT EXISTS "jobs_cloud_type_check"
    CHECK (cloud_type IN ('ALL', 'SECURE', 'COMMUNITY'));
