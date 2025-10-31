-- OpenWeights v1.0 Schema
-- This is a clean redesign with revocable API tokens

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

-- Create required extensions
-- Some extensions need explicit schemas, others are schema-independent
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "pgjwt" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";

-- Ensure public schema exists
CREATE SCHEMA IF NOT EXISTS "public";
ALTER SCHEMA "public" OWNER TO "pg_database_owner";
COMMENT ON SCHEMA "public" IS 'standard public schema';

-- =====================================================
-- ENUMS
-- =====================================================

CREATE TYPE "public"."job_status" AS ENUM (
    'pending',
    'in_progress',
    'completed',
    'failed',
    'canceled'
);

CREATE TYPE "public"."job_type" AS ENUM (
    'fine-tuning',
    'inference',
    'script',
    'api',
    'custom'
);

CREATE TYPE "public"."organization_role" AS ENUM (
    'admin',
    'user'
);

-- =====================================================
-- TABLES
-- =====================================================

-- Organizations
CREATE TABLE IF NOT EXISTS "public"."organizations" (
    "id" uuid DEFAULT gen_random_uuid() NOT NULL,
    "created_at" timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    "updated_at" timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    "name" text NOT NULL,
    PRIMARY KEY (id)
);

-- Organization Members
CREATE TABLE IF NOT EXISTS "public"."organization_members" (
    "organization_id" uuid NOT NULL,
    "user_id" uuid NOT NULL,
    "role" public.organization_role DEFAULT 'user'::public.organization_role NOT NULL,
    "created_at" timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    PRIMARY KEY (organization_id, user_id),
    FOREIGN KEY (organization_id) REFERENCES public.organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE
);

-- Organization Secrets (HF_TOKEN, RUNPOD_API_KEY, etc.)
CREATE TABLE IF NOT EXISTS "public"."organization_secrets" (
    "id" uuid DEFAULT gen_random_uuid() NOT NULL,
    "organization_id" uuid NOT NULL,
    "name" text NOT NULL,
    "value" text NOT NULL,
    "created_at" timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    "updated_at" timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (organization_id, name),
    FOREIGN KEY (organization_id) REFERENCES public.organizations(id) ON DELETE CASCADE
);

-- API Tokens (NEW: revocable tokens instead of JWTs as API keys)
CREATE TABLE IF NOT EXISTS "public"."api_tokens" (
    "id" uuid DEFAULT gen_random_uuid() NOT NULL,
    "organization_id" uuid NOT NULL,
    "name" text NOT NULL,
    "token_prefix" text NOT NULL, -- First 8 chars for display: "ow_abc12..."
    "token_hash" text NOT NULL,   -- SHA256 hash of full token
    "created_at" timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    "created_by" uuid NOT NULL,
    "last_used_at" timestamp with time zone,
    "expires_at" timestamp with time zone,
    "revoked_at" timestamp with time zone,
    PRIMARY KEY (id),
    FOREIGN KEY (organization_id) REFERENCES public.organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES auth.users(id) ON DELETE CASCADE
);

CREATE INDEX "api_tokens_token_hash_idx" ON "public"."api_tokens" USING btree (token_hash);
CREATE INDEX "api_tokens_organization_id_idx" ON "public"."api_tokens" USING btree (organization_id);

-- Files
CREATE TABLE IF NOT EXISTS "public"."files" (
    "id" text NOT NULL,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "filename" text NOT NULL,
    "purpose" text NOT NULL,
    "bytes" integer NOT NULL,
    "organization_id" uuid NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (organization_id) REFERENCES public.organizations(id) ON DELETE CASCADE
);

-- Workers
CREATE TABLE IF NOT EXISTS "public"."worker" (
    "id" text NOT NULL,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "status" text,
    "cached_models" text[],
    "vram_gb" integer,
    "pod_id" text,
    "ping" timestamp with time zone,
    "gpu_type" text,
    "gpu_count" integer,
    "docker_image" text,
    "organization_id" uuid NOT NULL,
    "logfile" text,
    "hardware_type" text,
    PRIMARY KEY (id),
    FOREIGN KEY (organization_id) REFERENCES public.organizations(id) ON DELETE CASCADE
);

-- Jobs
CREATE TABLE IF NOT EXISTS "public"."jobs" (
    "id" text NOT NULL,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "type" public.job_type NOT NULL,
    "model" text,
    "params" jsonb,
    "script" text,
    "requires_vram_gb" integer DEFAULT 24,
    "status" public.job_status DEFAULT 'pending'::public.job_status,
    "worker_id" text,
    "outputs" jsonb,
    "docker_image" text,
    "organization_id" uuid NOT NULL,
    "timeout" timestamp with time zone,
    "allowed_hardware" text[],
    PRIMARY KEY (id),
    FOREIGN KEY (organization_id) REFERENCES public.organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (worker_id) REFERENCES public.worker(id)
);

-- Runs
CREATE TABLE IF NOT EXISTS "public"."runs" (
    "id" serial NOT NULL,
    "job_id" text,
    "worker_id" text,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "status" public.job_status,
    "log_file" text,
    PRIMARY KEY (id),
    FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (worker_id) REFERENCES public.worker(id) ON DELETE SET NULL
);

-- Events
CREATE TABLE IF NOT EXISTS "public"."events" (
    "id" serial NOT NULL,
    "run_id" integer,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "data" jsonb NOT NULL,
    "file" text,
    PRIMARY KEY (id),
    FOREIGN KEY (run_id) REFERENCES public.runs(id) ON DELETE CASCADE
);

CREATE INDEX "events_run_id_idx" ON "public"."events" USING btree (run_id);

-- =====================================================
-- FUNCTIONS
-- =====================================================

-- Helper to get JWT secret from vault
CREATE OR REPLACE FUNCTION "public"."get_jwt_secret"()
RETURNS text
LANGUAGE plpgsql SECURITY DEFINER
AS $$
DECLARE
    secret text;
BEGIN
    SELECT decrypted_secret INTO secret
    FROM vault.decrypted_secrets
    WHERE name = 'jwt_secret'
    LIMIT 1;

    IF secret IS NULL THEN
        RAISE EXCEPTION 'JWT secret not found in vault';
    END IF;

    RETURN secret;
END;
$$;

-- Exchange API token for JWT
-- This is the NEW core function that replaces the old JWT-as-API-key approach
CREATE OR REPLACE FUNCTION "public"."exchange_api_token_for_jwt"(
    api_token text
)
RETURNS text
LANGUAGE plpgsql SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
    v_token_hash text;
    v_token_record record;
    v_jwt_secret text;
    v_jwt_token text;
BEGIN
    -- Hash the provided token
    v_token_hash := encode(extensions.digest(api_token, 'sha256'), 'hex');

    -- Look up token
    SELECT
        id,
        organization_id,
        expires_at,
        revoked_at
    INTO v_token_record
    FROM api_tokens
    WHERE token_hash = v_token_hash;

    -- Validate token exists
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Invalid API token';
    END IF;

    -- Check if revoked
    IF v_token_record.revoked_at IS NOT NULL THEN
        RAISE EXCEPTION 'API token has been revoked';
    END IF;

    -- Check if expired
    IF v_token_record.expires_at IS NOT NULL AND v_token_record.expires_at < now() THEN
        RAISE EXCEPTION 'API token has expired';
    END IF;

    -- Update last used timestamp
    UPDATE api_tokens
    SET last_used_at = now()
    WHERE id = v_token_record.id;

    -- Get JWT secret
    v_jwt_secret := get_jwt_secret();

    -- Create JWT with organization context
    v_jwt_token := extensions.sign(
        json_build_object(
            'role', 'authenticated',
            'iss', 'supabase',
            'iat', extract(epoch from now())::integer,
            'exp', extract(epoch from now() + interval '1 hour')::integer,
            'organization_id', v_token_record.organization_id,
            'api_token_id', v_token_record.id
        )::json,
        v_jwt_secret
    );

    RETURN v_jwt_token;
END;
$$;

-- Get organization ID from JWT claims
-- This works with the new JWT format (which still has organization_id in claims)
CREATE OR REPLACE FUNCTION "public"."get_organization_from_token"()
RETURNS uuid
LANGUAGE plpgsql STABLE
SET search_path TO 'public'
AS $$
DECLARE
    org_id uuid;
BEGIN
    -- Get org from JWT claims
    org_id := (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid;

    IF org_id IS NULL THEN
        RAISE EXCEPTION 'No organization_id in token claims';
    END IF;

    RETURN org_id;
END;
$$;

-- Check if user is organization member
CREATE OR REPLACE FUNCTION "public"."is_organization_member"(org_id uuid)
RETURNS boolean
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
    -- Check if the JWT has an organization_id claim that matches
    -- (for API token-based auth)
    IF (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid = org_id THEN
        RETURN TRUE;
    END IF;

    -- Otherwise check normal user membership
    RETURN EXISTS (
        SELECT 1
        FROM public.organization_members
        WHERE organization_id = org_id
          AND user_id = auth.uid()
    );
END;
$$;

-- Check if user is organization admin
CREATE OR REPLACE FUNCTION "public"."is_organization_admin"(org_id uuid)
RETURNS boolean
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
    -- API tokens have admin access to their organization
    IF (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid = org_id THEN
        RETURN TRUE;
    END IF;

    -- Otherwise check normal admin membership
    RETURN EXISTS (
        SELECT 1
        FROM public.organization_members
        WHERE organization_id = org_id
          AND user_id = auth.uid()
          AND role = 'admin'
    );
END;
$$;

-- Check if user has organization access (member or API token)
CREATE OR REPLACE FUNCTION "public"."has_organization_access"(org_id uuid)
RETURNS boolean
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
    RETURN is_organization_member(org_id);
END;
$$;

-- Create organization
CREATE OR REPLACE FUNCTION "public"."create_organization"(org_name text)
RETURNS uuid
LANGUAGE plpgsql SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
    v_org_id uuid;
BEGIN
    -- Check if authenticated
    IF auth.uid() IS NULL THEN
        RAISE EXCEPTION 'Authentication required';
    END IF;

    -- Create organization
    INSERT INTO organizations (name)
    VALUES (org_name)
    RETURNING id INTO v_org_id;

    -- Add creator as admin
    INSERT INTO organization_members (organization_id, user_id, role)
    VALUES (v_org_id, auth.uid(), 'admin');

    RETURN v_org_id;
END;
$$;

-- Create API token
-- Returns the full token (only time it's shown in plaintext)
CREATE OR REPLACE FUNCTION "public"."create_api_token"(
    org_id uuid,
    token_name text,
    expires_at timestamp with time zone DEFAULT NULL
)
RETURNS TABLE(token_id uuid, token text)
LANGUAGE plpgsql SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
    v_token_id uuid;
    v_token text;
    v_token_hash text;
    v_token_prefix text;
    v_created_by uuid;
    v_api_token_id uuid;
BEGIN
    -- Check if user is admin
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can create API tokens';
    END IF;

    -- Generate random token: ow_<48 random hex chars>
    v_token := 'ow_' || encode(extensions.gen_random_bytes(24), 'hex');
    v_token_hash := encode(extensions.digest(v_token, 'sha256'), 'hex');
    v_token_prefix := substring(v_token, 1, 11); -- "ow_" + first 8 hex chars

    -- Determine created_by: either from auth.uid() or from parent API token
    v_created_by := auth.uid();

    IF v_created_by IS NULL THEN
        -- If no user session, we're authenticated via API token
        -- Get the parent token's created_by
        v_api_token_id := (current_setting('request.jwt.claims', true)::json->>'api_token_id')::uuid;

        IF v_api_token_id IS NOT NULL THEN
            SELECT created_by INTO v_created_by
            FROM api_tokens
            WHERE id = v_api_token_id;
        END IF;

        IF v_created_by IS NULL THEN
            RAISE EXCEPTION 'Could not determine creator';
        END IF;
    END IF;

    -- Insert token
    INSERT INTO api_tokens (
        organization_id,
        name,
        token_prefix,
        token_hash,
        created_by,
        expires_at
    ) VALUES (
        org_id,
        token_name,
        v_token_prefix,
        v_token_hash,
        v_created_by,
        expires_at
    ) RETURNING id INTO v_token_id;

    -- Return token (ONLY time it's visible in plaintext!)
    RETURN QUERY SELECT v_token_id, v_token;
END;
$$;

-- Revoke API token
CREATE OR REPLACE FUNCTION "public"."revoke_api_token"(token_id uuid)
RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
    v_org_id uuid;
BEGIN
    -- Get organization for this token
    SELECT organization_id INTO v_org_id
    FROM api_tokens
    WHERE id = token_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Token not found';
    END IF;

    -- Check if user is admin
    IF NOT is_organization_admin(v_org_id) THEN
        RAISE EXCEPTION 'Only organization admins can revoke tokens';
    END IF;

    -- Revoke token
    UPDATE api_tokens
    SET revoked_at = now()
    WHERE id = token_id;

    RETURN TRUE;
END;
$$;

-- Organization management functions
CREATE OR REPLACE FUNCTION "public"."update_organization"(org_id uuid, new_name text)
RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can update organization';
    END IF;

    UPDATE organizations
    SET name = new_name
    WHERE id = org_id;

    RETURN found;
END;
$$;

CREATE OR REPLACE FUNCTION "public"."get_organization_members"(org_id uuid)
RETURNS TABLE(user_id uuid, email varchar, role public.organization_role)
LANGUAGE plpgsql SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
    RETURN QUERY
    SELECT
        om.user_id,
        au.email::varchar(255),
        om.role
    FROM public.organization_members om
    JOIN auth.users au ON au.id = om.user_id
    WHERE om.organization_id = org_id
    AND EXISTS (
        SELECT 1
        FROM public.organization_members viewer
        WHERE viewer.organization_id = org_id
        AND viewer.user_id = auth.uid()
    );
END;
$$;

CREATE OR REPLACE FUNCTION "public"."invite_organization_member"(
    org_id uuid,
    member_email varchar,
    member_role public.organization_role
)
RETURNS TABLE(user_id uuid, email varchar, role public.organization_role)
LANGUAGE plpgsql SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
    v_user_id uuid;
    v_email varchar(255);
BEGIN
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can invite members';
    END IF;

    SELECT au.id, au.email::varchar(255)
    INTO v_user_id, v_email
    FROM auth.users au
    WHERE lower(au.email) = lower(member_email);

    IF v_user_id IS NULL THEN
        RAISE EXCEPTION 'User with email % not found', member_email;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM organization_members om
        WHERE om.organization_id = org_id
        AND om.user_id = v_user_id
    ) THEN
        RAISE EXCEPTION 'User is already a member of this organization';
    END IF;

    INSERT INTO organization_members (organization_id, user_id, role)
    VALUES (org_id, v_user_id, member_role);

    RETURN QUERY
    SELECT v_user_id, v_email, member_role;
END;
$$;

CREATE OR REPLACE FUNCTION "public"."remove_organization_member"(org_id uuid, member_id uuid)
RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
    v_member_role organization_role;
    v_admin_count integer;
BEGIN
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can remove members';
    END IF;

    SELECT role INTO v_member_role
    FROM organization_members
    WHERE organization_id = org_id AND user_id = member_id;

    IF v_member_role = 'admin' THEN
        SELECT count(*) INTO v_admin_count
        FROM organization_members
        WHERE organization_id = org_id AND role = 'admin';

        IF v_admin_count <= 1 THEN
            RAISE EXCEPTION 'Cannot remove the last admin';
        END IF;
    END IF;

    DELETE FROM organization_members
    WHERE organization_id = org_id AND user_id = member_id;

    RETURN found;
END;
$$;

-- Secrets management
CREATE OR REPLACE FUNCTION "public"."manage_organization_secret"(
    org_id uuid,
    secret_name text,
    secret_value text
)
RETURNS uuid
LANGUAGE plpgsql SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
    v_secret_id uuid;
BEGIN
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can manage secrets';
    END IF;

    INSERT INTO organization_secrets (organization_id, name, value)
    VALUES (org_id, secret_name, secret_value)
    ON CONFLICT (organization_id, name)
    DO UPDATE SET value = excluded.value, updated_at = now()
    RETURNING id INTO v_secret_id;

    RETURN v_secret_id;
END;
$$;

CREATE OR REPLACE FUNCTION "public"."delete_organization_secret"(org_id uuid, secret_name text)
RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER
SET search_path TO 'public'
AS $$
BEGIN
    IF NOT is_organization_admin(org_id) THEN
        RAISE EXCEPTION 'Only organization admins can manage secrets';
    END IF;

    DELETE FROM organization_secrets
    WHERE organization_id = org_id AND name = secret_name;

    RETURN found;
END;
$$;

-- Job management functions
CREATE OR REPLACE FUNCTION "public"."acquire_job"(_job_id text, _worker_id text)
RETURNS SETOF public.jobs
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    UPDATE jobs
    SET status = 'in_progress',
        worker_id = _worker_id
    WHERE id = _job_id
      AND status = 'pending'
    RETURNING *;
END;
$$;

CREATE OR REPLACE FUNCTION "public"."update_job_status_if_in_progress"(
    _job_id text,
    _new_status text,
    _worker_id text,
    _job_outputs jsonb DEFAULT NULL,
    _job_script text DEFAULT NULL
)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE jobs
    SET status = _new_status::job_status,
        outputs = _job_outputs,
        script = COALESCE(_job_script, script)
    WHERE id = _job_id
      AND status = 'in_progress'
      AND worker_id = _worker_id;
END;
$$;

-- Hardware matching
CREATE OR REPLACE FUNCTION "public"."hardware_matches"(
    worker_hardware text,
    allowed_hardware text[]
)
RETURNS boolean
LANGUAGE plpgsql
AS $$
BEGIN
    IF allowed_hardware IS NULL OR array_length(allowed_hardware, 1) IS NULL THEN
        RETURN TRUE;
    END IF;

    RETURN worker_hardware = ANY(allowed_hardware);
END;
$$;

-- Storage path helper
CREATE OR REPLACE FUNCTION "public"."get_path_organization_id"(path text)
RETURNS uuid
LANGUAGE plpgsql STABLE
AS $$
DECLARE
    parts text[];
    org_id uuid;
BEGIN
    parts := string_to_array(path, '/');

    IF array_length(parts, 1) IS NULL OR parts[1] <> 'organizations' THEN
        RETURN NULL;
    END IF;

    BEGIN
        org_id := parts[2]::uuid;
        RETURN org_id;
    EXCEPTION WHEN others THEN
        RETURN NULL;
    END;
END;
$$;

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION "public"."handle_updated_at"()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION "public"."set_deployment_timeout"()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.type = 'api' THEN
        NEW.timeout = NEW.created_at + interval '1 hour';
    END IF;
    RETURN NEW;
END;
$$;

-- Create triggers
CREATE TRIGGER set_updated_at_organizations
    BEFORE UPDATE ON public.organizations
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER set_updated_at_organization_secrets
    BEFORE UPDATE ON public.organization_secrets
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER set_updated_at_jobs
    BEFORE UPDATE ON public.jobs
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER set_updated_at_runs
    BEFORE UPDATE ON public.runs
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER set_updated_at_worker
    BEFORE UPDATE ON public.worker
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER set_deployment_timeout_trigger
    BEFORE INSERT ON public.jobs
    FOR EACH ROW EXECUTE FUNCTION public.set_deployment_timeout();

-- =====================================================
-- ROW LEVEL SECURITY POLICIES
-- =====================================================

ALTER TABLE public.organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organization_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organization_secrets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.worker ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.events ENABLE ROW LEVEL SECURITY;

-- Organizations
CREATE POLICY "Enable read access for organization members"
    ON public.organizations FOR SELECT
    USING (public.is_organization_member(id));

CREATE POLICY "Enable write access for organization admins"
    ON public.organizations
    USING (public.is_organization_admin(id));

-- Organization Members
CREATE POLICY "Enable read access for members"
    ON public.organization_members FOR SELECT
    USING (public.is_organization_member(organization_id));

CREATE POLICY "Enable write access for admins"
    ON public.organization_members
    USING (public.is_organization_admin(organization_id));

-- Organization Secrets
CREATE POLICY "Organization admins can manage secrets"
    ON public.organization_secrets
    USING (public.is_organization_admin(organization_id));

-- API Tokens
CREATE POLICY "Organization admins can manage tokens"
    ON public.api_tokens
    USING (public.is_organization_admin(organization_id));

CREATE POLICY "Organization members can view their tokens"
    ON public.api_tokens FOR SELECT
    USING (public.is_organization_member(organization_id));

-- Files
CREATE POLICY "Organization members can access files"
    ON public.files
    USING (public.is_organization_member(organization_id));

-- Workers
CREATE POLICY "Enable access for organization members"
    ON public.worker
    USING (public.is_organization_member(organization_id));

-- Jobs
CREATE POLICY "Organization members can read jobs"
    ON public.jobs FOR SELECT
    USING (public.is_organization_member(organization_id));

CREATE POLICY "Organization members can insert jobs"
    ON public.jobs FOR INSERT
    WITH CHECK (organization_id = public.get_organization_from_token());

CREATE POLICY "Organization members can update their jobs"
    ON public.jobs FOR UPDATE
    USING (public.is_organization_member(organization_id));

CREATE POLICY "Organization members can delete their jobs"
    ON public.jobs FOR DELETE
    USING (public.is_organization_member(organization_id));

-- Runs
CREATE POLICY "Enable access for organization members"
    ON public.runs
    USING (EXISTS (
        SELECT 1
        FROM public.jobs j
        WHERE j.id = runs.job_id
          AND public.is_organization_member(j.organization_id)
    ));

-- Events
CREATE POLICY "Enable access for organization members"
    ON public.events
    USING (EXISTS (
        SELECT 1
        FROM public.runs r
        JOIN public.jobs j ON j.id = r.job_id
        WHERE r.id = events.run_id
          AND public.is_organization_member(j.organization_id)
    ));

-- =====================================================
-- GRANTS
-- =====================================================

GRANT USAGE ON SCHEMA public TO postgres;
GRANT USAGE ON SCHEMA public TO anon;
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT USAGE ON SCHEMA public TO service_role;

GRANT ALL ON ALL TABLES IN SCHEMA public TO anon;
GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;

GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;

GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO anon;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO authenticated;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO service_role;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON SEQUENCES TO postgres;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON SEQUENCES TO anon;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON SEQUENCES TO authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON SEQUENCES TO service_role;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON FUNCTIONS TO postgres;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON FUNCTIONS TO anon;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON FUNCTIONS TO authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON FUNCTIONS TO service_role;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON TABLES TO anon;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON TABLES TO authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON TABLES TO service_role;
