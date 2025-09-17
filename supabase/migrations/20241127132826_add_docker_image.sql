-- Add docker_image column to jobs table
ALTER TABLE "public"."jobs"
    ADD COLUMN "docker_image" text;

-- Grant permissions to match existing table permissions
GRANT ALL ON TABLE "public"."jobs" TO "anon";
GRANT ALL ON TABLE "public"."jobs" TO "authenticated";
GRANT ALL ON TABLE "public"."jobs" TO "service_role";
