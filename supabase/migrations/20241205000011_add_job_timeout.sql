-- Add timeout column to jobs table
ALTER TABLE "public"."jobs"
    ADD COLUMN "timeout" timestamp with time zone;

-- Function to set initial timeout for deployment jobs
CREATE OR REPLACE FUNCTION set_deployment_timeout()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.type = 'api' THEN
        NEW.timeout = NEW.created_at + interval '1 hour';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for new deployment jobs
CREATE TRIGGER set_deployment_timeout_trigger
    BEFORE INSERT ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION set_deployment_timeout();
