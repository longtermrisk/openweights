-- Add gpu_type and gpu_count columns to worker table
ALTER TABLE "public"."worker"
    ADD COLUMN "gpu_type" text,
    ADD COLUMN "gpu_count" integer;

-- Grant permissions to match existing table permissions
GRANT ALL ON TABLE "public"."worker" TO "anon";
GRANT ALL ON TABLE "public"."worker" TO "authenticated";
GRANT ALL ON TABLE "public"."worker" TO "service_role";
