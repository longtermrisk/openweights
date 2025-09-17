-- Add updated_at columns to tables
ALTER TABLE "public"."worker"
    ADD COLUMN "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP;

ALTER TABLE "public"."jobs"
    ADD COLUMN "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP;

ALTER TABLE "public"."runs"
    ADD COLUMN "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP;

-- Update existing rows to set updated_at = created_at
UPDATE "public"."worker" SET "updated_at" = "created_at";
UPDATE "public"."jobs" SET "updated_at" = "created_at";
UPDATE "public"."runs" SET "updated_at" = "created_at";

-- Create a function to automatically set updated_at
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for each table
CREATE TRIGGER set_updated_at_worker
    BEFORE UPDATE ON public.worker
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER set_updated_at_jobs
    BEFORE UPDATE ON public.jobs
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER set_updated_at_runs
    BEFORE UPDATE ON public.runs
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_updated_at();

-- Grant permissions to match existing table permissions
GRANT ALL ON TABLE "public"."worker" TO "anon";
GRANT ALL ON TABLE "public"."worker" TO "authenticated";
GRANT ALL ON TABLE "public"."worker" TO "service_role";

GRANT ALL ON TABLE "public"."jobs" TO "anon";
GRANT ALL ON TABLE "public"."jobs" TO "authenticated";
GRANT ALL ON TABLE "public"."jobs" TO "service_role";

GRANT ALL ON TABLE "public"."runs" TO "anon";
GRANT ALL ON TABLE "public"."runs" TO "authenticated";
GRANT ALL ON TABLE "public"."runs" TO "service_role";
