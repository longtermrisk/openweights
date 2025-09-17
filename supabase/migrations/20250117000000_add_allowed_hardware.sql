-- Add allowed_hardware column to jobs table
ALTER TABLE "public"."jobs"
    ADD COLUMN "allowed_hardware" text[] DEFAULT NULL;

-- Add hardware_type column to worker table
-- This will store the actual hardware configuration of the worker (e.g., '2x A100')
ALTER TABLE "public"."worker"
    ADD COLUMN "hardware_type" text DEFAULT NULL;

-- Grant permissions to match existing table permissions
GRANT ALL ON TABLE "public"."jobs" TO "anon";
GRANT ALL ON TABLE "public"."jobs" TO "authenticated";
GRANT ALL ON TABLE "public"."jobs" TO "service_role";

GRANT ALL ON TABLE "public"."worker" TO "anon";
GRANT ALL ON TABLE "public"."worker" TO "authenticated";
GRANT ALL ON TABLE "public"."worker" TO "service_role";

-- Create a function to check if a worker's hardware matches the job's allowed hardware
CREATE OR REPLACE FUNCTION hardware_matches(
    worker_hardware text,
    allowed_hardware text[]
)
RETURNS boolean
LANGUAGE plpgsql
AS $$
BEGIN
    -- If allowed_hardware is NULL or empty array, any hardware is allowed
    IF allowed_hardware IS NULL OR array_length(allowed_hardware, 1) IS NULL THEN
        RETURN TRUE;
    END IF;

    -- Check if worker's hardware is in the allowed list
    RETURN worker_hardware = ANY(allowed_hardware);
END;
$$;

-- Grant execute permissions
GRANT EXECUTE ON FUNCTION hardware_matches(text, text[]) TO "anon";
GRANT EXECUTE ON FUNCTION hardware_matches(text, text[]) TO "authenticated";
GRANT EXECUTE ON FUNCTION hardware_matches(text, text[]) TO "service_role";
