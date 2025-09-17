-- Add logfile column to worker table
ALTER TABLE "public"."worker"
    ADD COLUMN "logfile" text;

-- Grant permissions to match existing table permissions
GRANT ALL ON TABLE "public"."worker" TO "anon";
GRANT ALL ON TABLE "public"."worker" TO "authenticated";
GRANT ALL ON TABLE "public"."worker" TO "service_role";
