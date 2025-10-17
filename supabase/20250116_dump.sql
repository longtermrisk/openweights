

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
CREATE EXTENSION IF NOT EXISTS "pgsodium" WITH SCHEMA "pgsodium";
CREATE EXTENSION IF NOT EXISTS "pg_graphql" WITH SCHEMA "graphql";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "pgjwt" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";

CREATE SCHEMA IF NOT EXISTS "app_auth";


ALTER SCHEMA "app_auth" OWNER TO "postgres";


CREATE SCHEMA IF NOT EXISTS "public";


ALTER SCHEMA "public" OWNER TO "pg_database_owner";


COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE TYPE "public"."job_status" AS ENUM (
    'pending',
    'in_progress',
    'completed',
    'failed',
    'canceled'
);


ALTER TYPE "public"."job_status" OWNER TO "postgres";


CREATE TYPE "public"."job_type" AS ENUM (
    'fine-tuning',
    'inference',
    'script',
    'api',
    'custom'
);


ALTER TYPE "public"."job_type" OWNER TO "postgres";


CREATE TYPE "public"."organization_role" AS ENUM (
    'admin',
    'user'
);


ALTER TYPE "public"."organization_role" OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "app_auth"."check_if_service_account"() RETURNS boolean
    LANGUAGE "sql" SECURITY DEFINER
    AS $$
  select coalesce(
    current_setting('request.jwt.claims', true)::json->>'is_service_account',
    'false'
  )::boolean;
$$;


ALTER FUNCTION "app_auth"."check_if_service_account"() OWNER TO "postgres";

SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."jobs" (
    "id" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "type" "public"."job_type" NOT NULL,
    "model" "text",
    "params" "jsonb",
    "script" "text",
    "requires_vram_gb" integer DEFAULT 24,
    "status" "public"."job_status" DEFAULT 'pending'::"public"."job_status",
    "worker_id" "text",
    "outputs" "jsonb",
    "docker_image" "text",
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "organization_id" "uuid" NOT NULL,
    "timeout" timestamp with time zone,
    "allowed_hardware" "text"[]
);


ALTER TABLE "public"."jobs" OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."acquire_job"("_job_id" "text", "_worker_id" "text") RETURNS SETOF "public"."jobs"
    LANGUAGE "plpgsql"
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


ALTER FUNCTION "public"."acquire_job"("_job_id" "text", "_worker_id" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."create_organization"("org_name" "text") RETURNS "uuid"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
declare
    v_org_id uuid;
begin
    -- Check if authenticated
    if auth.uid() is null then
        raise exception 'Authentication required';
    end if;

    -- Create organization
    insert into organizations (name)
    values (org_name)
    returning id into v_org_id;

    -- Add creator as admin
    insert into organization_members (organization_id, user_id, role)
    values (v_org_id, auth.uid(), 'admin');

    return v_org_id;
end;
$$;


ALTER FUNCTION "public"."create_organization"("org_name" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."create_service_account_token"("org_id" "uuid", "token_name" "text", "created_by" "uuid", "expires_at" timestamp with time zone DEFAULT NULL::timestamp with time zone) RETURNS TABLE("token_id" "uuid", "jwt_token" "text")
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
declare
    v_token_id uuid;
    v_jwt_secret text;
    v_jwt_token text;
begin
    -- Get JWT secret
    v_jwt_secret := get_jwt_secret();

    if v_jwt_secret is null then
        raise exception 'JWT secret not configured';
    end if;

    -- Create token record
    insert into tokens (organization_id, name, expires_at, created_by)
    values (org_id, token_name, expires_at, created_by)
    returning id into v_token_id;

    -- Create JWT token with service account claims
    v_jwt_token := extensions.sign(
        json_build_object(
            'role', 'authenticated',
            'iss', 'supabase',
            'iat', extract(epoch from now())::integer,
            'exp', case
                when expires_at is null then extract(epoch from now() + interval '10 years')::integer
                else extract(epoch from expires_at)::integer
            end,
            'is_service_account', true,
            'organization_id', org_id,
            'token_id', v_token_id
        )::json,
        v_jwt_secret
    );

    -- Update token hash
    update tokens
    set token_hash = v_jwt_token
    where id = v_token_id;

    return query select v_token_id, v_jwt_token;
end;
$$;


ALTER FUNCTION "public"."create_service_account_token"("org_id" "uuid", "token_name" "text", "created_by" "uuid", "expires_at" timestamp with time zone) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."delete_organization_secret"("org_id" "uuid", "secret_name" "text") RETURNS boolean
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
begin
    -- Check if user is admin
    if not is_organization_admin(org_id) then
        raise exception 'Only organization admins can manage secrets';
    end if;

    -- Delete secret
    delete from organization_secrets
    where organization_id = org_id and name = secret_name;

    return found;
end;
$$;


ALTER FUNCTION "public"."delete_organization_secret"("org_id" "uuid", "secret_name" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_jwt_secret"() RETURNS "text"
    LANGUAGE "plpgsql" SECURITY DEFINER
    AS $$
declare
    secret text;
begin
    -- Get secret from vault
    select decrypted_secret into secret
    from vault.decrypted_secrets
    where name = 'jwt_secret'
    limit 1;

    if secret is null then
        raise exception 'JWT secret not found in vault';
    end if;

    return secret;
end;
$$;


ALTER FUNCTION "public"."get_jwt_secret"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_organization_from_token"() RETURNS "uuid"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public', 'auth', 'app_auth'
    AS $$
DECLARE
  org_id uuid;
BEGIN
  -- Only handle service account tokens
  IF NOT app_auth.check_if_service_account() THEN
    RAISE EXCEPTION 'Only service account tokens are supported';
  END IF;

  -- Get org from claims
  org_id := (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid;

  -- Update last_used_at in tokens table
  UPDATE public.tokens
  SET last_used_at = now()
  WHERE id = (current_setting('request.jwt.claims', true)::json->>'token_id')::uuid;

  RETURN org_id;
END;
$$;


ALTER FUNCTION "public"."get_organization_from_token"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_organization_members"("org_id" "uuid") RETURNS TABLE("user_id" "uuid", "email" character varying, "role" "public"."organization_role")
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
begin
    return query
    select
        om.user_id,
        au.email::varchar(255),  -- Explicitly cast email to match return type
        om.role
    from public.organization_members om
    join auth.users au on au.id = om.user_id
    where om.organization_id = org_id
    and exists (
        select 1
        from public.organization_members viewer
        where viewer.organization_id = org_id
        and viewer.user_id = auth.uid()
    );
end;
$$;


ALTER FUNCTION "public"."get_organization_members"("org_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_path_organization_id"("path" "text") RETURNS "uuid"
    LANGUAGE "plpgsql" STABLE
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


ALTER FUNCTION "public"."get_path_organization_id"("path" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."handle_updated_at"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."handle_updated_at"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."hardware_matches"("worker_hardware" "text", "allowed_hardware" "text"[]) RETURNS boolean
    LANGUAGE "plpgsql"
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


ALTER FUNCTION "public"."hardware_matches"("worker_hardware" "text", "allowed_hardware" "text"[]) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."has_organization_access"("org_id" "uuid") RETURNS boolean
    LANGUAGE "plpgsql" STABLE SECURITY DEFINER
    SET "search_path" TO 'public', 'auth', 'app_auth'
    AS $$
BEGIN
  -- Service account?
  IF app_auth.check_if_service_account() THEN
    RETURN (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid = org_id;
  END IF;

  -- Otherwise, membership
  RETURN EXISTS (
    SELECT 1
    FROM public.organization_members
    WHERE organization_id = org_id
      AND user_id = auth.uid()
  );
END;
$$;


ALTER FUNCTION "public"."has_organization_access"("org_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."invite_organization_member"("org_id" "uuid", "member_email" character varying, "member_role" "public"."organization_role") RETURNS TABLE("user_id" "uuid", "email" character varying, "role" "public"."organization_role")
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
declare
    v_user_id uuid;
    v_email varchar(255);
begin
    -- Check if the inviter is an admin
    if not is_organization_admin(org_id) then
        raise exception 'Only organization admins can invite members';
    end if;

    -- Get the user ID for the email
    select au.id, au.email::varchar(255)
    into v_user_id, v_email
    from auth.users au
    where lower(au.email) = lower(member_email);

    if v_user_id is null then
        raise exception 'User with email % not found', member_email;
    end if;

    -- Check if user is already a member
    if exists (
        select 1
        from organization_members om
        where om.organization_id = org_id
        and om.user_id = v_user_id
    ) then
        raise exception 'User is already a member of this organization';
    end if;

    -- Insert the new member
    insert into organization_members (organization_id, user_id, role)
    values (org_id, v_user_id, member_role);

    -- Return the result
    return query
    select
        v_user_id as user_id,
        v_email as email,
        member_role as role;
end;
$$;


ALTER FUNCTION "public"."invite_organization_member"("org_id" "uuid", "member_email" character varying, "member_role" "public"."organization_role") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."is_organization_admin"("org_id" "uuid") RETURNS boolean
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public', 'auth', 'app_auth'
    AS $$
BEGIN
  -- Service accounts have admin access to their organization
  IF app_auth.check_if_service_account() THEN
    RETURN (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid = org_id;
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


ALTER FUNCTION "public"."is_organization_admin"("org_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."is_organization_member"("org_id" "uuid") RETURNS boolean
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public', 'auth', 'app_auth'
    AS $$
BEGIN
  -- If this is a service account, check the organization claim
  IF app_auth.check_if_service_account() THEN
    RETURN (current_setting('request.jwt.claims', true)::json->>'organization_id')::uuid = org_id;
  END IF;

  -- Otherwise check normal membership
  RETURN EXISTS (
    SELECT 1
    FROM public.organization_members
    WHERE organization_id = org_id
      AND user_id = auth.uid()
  );
END;
$$;


ALTER FUNCTION "public"."is_organization_member"("org_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."manage_organization_secret"("org_id" "uuid", "secret_name" "text", "secret_value" "text") RETURNS "uuid"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
declare
    v_secret_id uuid;
begin
    -- Check if user is admin
    if not is_organization_admin(org_id) then
        raise exception 'Only organization admins can manage secrets';
    end if;

    -- Insert or update secret
    insert into organization_secrets (organization_id, name, value)
    values (org_id, secret_name, secret_value)
    on conflict (organization_id, name)
    do update set value = excluded.value, updated_at = now()
    returning id into v_secret_id;

    return v_secret_id;
end;
$$;


ALTER FUNCTION "public"."manage_organization_secret"("org_id" "uuid", "secret_name" "text", "secret_value" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."remove_organization_member"("org_id" "uuid", "member_id" "uuid") RETURNS boolean
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
declare
    v_member_role organization_role;
    v_admin_count integer;
begin
    -- Check if user is admin
    if not is_organization_admin(org_id) then
        raise exception 'Only organization admins can remove members';
    end if;

    -- Get member's role
    select role into v_member_role
    from organization_members
    where organization_id = org_id and user_id = member_id;

    -- If member is admin, check if they're not the last admin
    if v_member_role = 'admin' then
        select count(*) into v_admin_count
        from organization_members
        where organization_id = org_id and role = 'admin';

        if v_admin_count <= 1 then
            raise exception 'Cannot remove the last admin';
        end if;
    end if;

    -- Remove member
    delete from organization_members
    where organization_id = org_id and user_id = member_id;

    return found;
end;
$$;


ALTER FUNCTION "public"."remove_organization_member"("org_id" "uuid", "member_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."set_deployment_timeout"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    IF NEW.type = 'api' THEN
        NEW.timeout = NEW.created_at + interval '1 hour';
    END IF;
    RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."set_deployment_timeout"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."update_job_status_if_in_progress"("_job_id" "text", "_new_status" "text", "_worker_id" "text", "_job_outputs" "jsonb" DEFAULT NULL::"jsonb", "_job_script" "text" DEFAULT NULL::"text") RETURNS "void"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  UPDATE jobs
     SET status = _new_status::job_status,   -- <--- cast to job_status
         outputs = _job_outputs,
         script = COALESCE(_job_script, script)
   WHERE id = _job_id
     AND status = 'in_progress'   -- Assuming 'in_progress' also exists in your enum
     AND worker_id = _worker_id;
END;
$$;


ALTER FUNCTION "public"."update_job_status_if_in_progress"("_job_id" "text", "_new_status" "text", "_worker_id" "text", "_job_outputs" "jsonb", "_job_script" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."update_organization"("org_id" "uuid", "new_name" "text") RETURNS boolean
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
begin
    -- Check if user is admin
    if not is_organization_admin(org_id) then
        raise exception 'Only organization admins can update organization';
    end if;

    -- Update organization
    update organizations
    set name = new_name
    where id = org_id;

    return found;
end;
$$;


ALTER FUNCTION "public"."update_organization"("org_id" "uuid", "new_name" "text") OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."events" (
    "id" integer NOT NULL,
    "run_id" integer,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "data" "jsonb" NOT NULL,
    "file" "text"
);


ALTER TABLE "public"."events" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."events_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."events_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."events_id_seq" OWNED BY "public"."events"."id";



CREATE TABLE IF NOT EXISTS "public"."files" (
    "id" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "filename" "text" NOT NULL,
    "purpose" "text" NOT NULL,
    "bytes" integer NOT NULL
);


ALTER TABLE "public"."files" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."organization_members" (
    "organization_id" "uuid" NOT NULL,
    "user_id" "uuid" NOT NULL,
    "role" "public"."organization_role" DEFAULT 'user'::"public"."organization_role" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "timezone"('utc'::"text", "now"()) NOT NULL
);


ALTER TABLE "public"."organization_members" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."organization_secrets" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "organization_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "value" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "timezone"('utc'::"text", "now"()) NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "timezone"('utc'::"text", "now"()) NOT NULL
);


ALTER TABLE "public"."organization_secrets" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."organizations" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "created_at" timestamp with time zone DEFAULT "timezone"('utc'::"text", "now"()) NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "timezone"('utc'::"text", "now"()) NOT NULL,
    "name" "text" NOT NULL
);


ALTER TABLE "public"."organizations" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."runs" (
    "id" integer NOT NULL,
    "job_id" "text",
    "worker_id" "text",
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "status" "public"."job_status",
    "log_file" "text",
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE "public"."runs" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."runs_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."runs_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."runs_id_seq" OWNED BY "public"."runs"."id";



CREATE TABLE IF NOT EXISTS "public"."tokens" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "organization_id" "uuid" NOT NULL,
    "name" "text" NOT NULL,
    "token_hash" "text",
    "expires_at" timestamp with time zone,
    "created_at" timestamp with time zone DEFAULT "timezone"('utc'::"text", "now"()) NOT NULL,
    "created_by" "uuid" NOT NULL,
    "last_used_at" timestamp with time zone
);


ALTER TABLE "public"."tokens" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."worker" (
    "id" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "status" "text",
    "cached_models" "text"[],
    "vram_gb" integer,
    "pod_id" "text",
    "ping" timestamp with time zone,
    "gpu_type" "text",
    "gpu_count" integer,
    "docker_image" "text",
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "organization_id" "uuid" NOT NULL,
    "logfile" "text",
    "hardware_type" "text"
);


ALTER TABLE "public"."worker" OWNER TO "postgres";


ALTER TABLE ONLY "public"."events" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."events_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."runs" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."runs_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."events"
    ADD CONSTRAINT "events_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."files"
    ADD CONSTRAINT "files_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."jobs"
    ADD CONSTRAINT "jobs_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."organization_members"
    ADD CONSTRAINT "organization_members_pkey" PRIMARY KEY ("organization_id", "user_id");



ALTER TABLE ONLY "public"."organization_secrets"
    ADD CONSTRAINT "organization_secrets_organization_id_name_key" UNIQUE ("organization_id", "name");



ALTER TABLE ONLY "public"."organization_secrets"
    ADD CONSTRAINT "organization_secrets_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."organizations"
    ADD CONSTRAINT "organizations_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."runs"
    ADD CONSTRAINT "runs_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."tokens"
    ADD CONSTRAINT "tokens_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."worker"
    ADD CONSTRAINT "worker_pkey" PRIMARY KEY ("id");



CREATE INDEX "events_run_id_idx" ON "public"."events" USING "btree" ("run_id");



CREATE OR REPLACE TRIGGER "set_deployment_timeout_trigger" BEFORE INSERT ON "public"."jobs" FOR EACH ROW EXECUTE FUNCTION "public"."set_deployment_timeout"();



CREATE OR REPLACE TRIGGER "set_updated_at_jobs" BEFORE UPDATE ON "public"."jobs" FOR EACH ROW EXECUTE FUNCTION "public"."handle_updated_at"();



CREATE OR REPLACE TRIGGER "set_updated_at_organization_secrets" BEFORE UPDATE ON "public"."organization_secrets" FOR EACH ROW EXECUTE FUNCTION "public"."handle_updated_at"();



CREATE OR REPLACE TRIGGER "set_updated_at_organizations" BEFORE UPDATE ON "public"."organizations" FOR EACH ROW EXECUTE FUNCTION "public"."handle_updated_at"();



CREATE OR REPLACE TRIGGER "set_updated_at_runs" BEFORE UPDATE ON "public"."runs" FOR EACH ROW EXECUTE FUNCTION "public"."handle_updated_at"();



CREATE OR REPLACE TRIGGER "set_updated_at_worker" BEFORE UPDATE ON "public"."worker" FOR EACH ROW EXECUTE FUNCTION "public"."handle_updated_at"();



ALTER TABLE ONLY "public"."events"
    ADD CONSTRAINT "events_run_id_fkey" FOREIGN KEY ("run_id") REFERENCES "public"."runs"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."jobs"
    ADD CONSTRAINT "jobs_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."jobs"
    ADD CONSTRAINT "jobs_worker_id_fkey" FOREIGN KEY ("worker_id") REFERENCES "public"."worker"("id");



ALTER TABLE ONLY "public"."organization_members"
    ADD CONSTRAINT "organization_members_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."organization_members"
    ADD CONSTRAINT "organization_members_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."organization_secrets"
    ADD CONSTRAINT "organization_secrets_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."runs"
    ADD CONSTRAINT "runs_job_id_fkey" FOREIGN KEY ("job_id") REFERENCES "public"."jobs"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."runs"
    ADD CONSTRAINT "runs_worker_id_fkey" FOREIGN KEY ("worker_id") REFERENCES "public"."worker"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."tokens"
    ADD CONSTRAINT "tokens_created_by_fkey" FOREIGN KEY ("created_by") REFERENCES "auth"."users"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."tokens"
    ADD CONSTRAINT "tokens_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."worker"
    ADD CONSTRAINT "worker_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations"("id") ON DELETE CASCADE;



CREATE POLICY "Enable access for organization members" ON "public"."events" USING ((EXISTS ( SELECT 1
   FROM ("public"."runs" "r"
     JOIN "public"."jobs" "j" ON (("j"."id" = "r"."job_id")))
  WHERE (("r"."id" = "events"."run_id") AND "public"."is_organization_member"("j"."organization_id")))));



CREATE POLICY "Enable access for organization members" ON "public"."runs" USING ((EXISTS ( SELECT 1
   FROM "public"."jobs" "j"
  WHERE (("j"."id" = "runs"."job_id") AND "public"."is_organization_member"("j"."organization_id")))));



CREATE POLICY "Enable access for organization members" ON "public"."worker" USING ("public"."is_organization_member"("organization_id"));



CREATE POLICY "Enable read access for members" ON "public"."organization_members" FOR SELECT USING ("public"."is_organization_member"("organization_id"));



CREATE POLICY "Enable read access for organization members" ON "public"."organizations" FOR SELECT USING ("public"."is_organization_member"("id"));



CREATE POLICY "Enable write access for admins" ON "public"."organization_members" USING ("public"."is_organization_admin"("organization_id"));



CREATE POLICY "Enable write access for organization admins" ON "public"."organizations" USING ("public"."is_organization_admin"("id"));



CREATE POLICY "Organization admins can manage secrets" ON "public"."organization_secrets" USING ("public"."is_organization_admin"("organization_id"));



CREATE POLICY "Organization admins can manage tokens" ON "public"."tokens" USING ("public"."is_organization_admin"("organization_id"));



CREATE POLICY "Organization members can delete their jobs" ON "public"."jobs" FOR DELETE USING ("public"."is_organization_member"("organization_id"));



CREATE POLICY "Organization members can insert jobs" ON "public"."jobs" FOR INSERT WITH CHECK (("organization_id" = "public"."get_organization_from_token"()));



CREATE POLICY "Organization members can read jobs" ON "public"."jobs" FOR SELECT USING ("public"."is_organization_member"("organization_id"));



CREATE POLICY "Organization members can update their jobs" ON "public"."jobs" FOR UPDATE USING ("public"."is_organization_member"("organization_id"));



CREATE POLICY "Organization members can view their tokens" ON "public"."tokens" FOR SELECT USING ("public"."is_organization_member"("organization_id"));



ALTER TABLE "public"."events" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."jobs" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."organization_members" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."organization_secrets" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."organizations" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."runs" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."tokens" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."worker" ENABLE ROW LEVEL SECURITY;


GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";



GRANT ALL ON TABLE "public"."jobs" TO "anon";
GRANT ALL ON TABLE "public"."jobs" TO "authenticated";
GRANT ALL ON TABLE "public"."jobs" TO "service_role";



GRANT ALL ON FUNCTION "public"."acquire_job"("_job_id" "text", "_worker_id" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."acquire_job"("_job_id" "text", "_worker_id" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."acquire_job"("_job_id" "text", "_worker_id" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."create_organization"("org_name" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."create_organization"("org_name" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."create_organization"("org_name" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."create_service_account_token"("org_id" "uuid", "token_name" "text", "created_by" "uuid", "expires_at" timestamp with time zone) TO "anon";
GRANT ALL ON FUNCTION "public"."create_service_account_token"("org_id" "uuid", "token_name" "text", "created_by" "uuid", "expires_at" timestamp with time zone) TO "authenticated";
GRANT ALL ON FUNCTION "public"."create_service_account_token"("org_id" "uuid", "token_name" "text", "created_by" "uuid", "expires_at" timestamp with time zone) TO "service_role";



GRANT ALL ON FUNCTION "public"."delete_organization_secret"("org_id" "uuid", "secret_name" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."delete_organization_secret"("org_id" "uuid", "secret_name" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."delete_organization_secret"("org_id" "uuid", "secret_name" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."get_jwt_secret"() TO "anon";
GRANT ALL ON FUNCTION "public"."get_jwt_secret"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_jwt_secret"() TO "service_role";



GRANT ALL ON FUNCTION "public"."get_organization_from_token"() TO "anon";
GRANT ALL ON FUNCTION "public"."get_organization_from_token"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_organization_from_token"() TO "service_role";



GRANT ALL ON FUNCTION "public"."get_organization_members"("org_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."get_organization_members"("org_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_organization_members"("org_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."get_path_organization_id"("path" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."get_path_organization_id"("path" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_path_organization_id"("path" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."handle_updated_at"() TO "anon";
GRANT ALL ON FUNCTION "public"."handle_updated_at"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."handle_updated_at"() TO "service_role";



GRANT ALL ON FUNCTION "public"."hardware_matches"("worker_hardware" "text", "allowed_hardware" "text"[]) TO "anon";
GRANT ALL ON FUNCTION "public"."hardware_matches"("worker_hardware" "text", "allowed_hardware" "text"[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."hardware_matches"("worker_hardware" "text", "allowed_hardware" "text"[]) TO "service_role";



GRANT ALL ON FUNCTION "public"."has_organization_access"("org_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."has_organization_access"("org_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."has_organization_access"("org_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."invite_organization_member"("org_id" "uuid", "member_email" character varying, "member_role" "public"."organization_role") TO "anon";
GRANT ALL ON FUNCTION "public"."invite_organization_member"("org_id" "uuid", "member_email" character varying, "member_role" "public"."organization_role") TO "authenticated";
GRANT ALL ON FUNCTION "public"."invite_organization_member"("org_id" "uuid", "member_email" character varying, "member_role" "public"."organization_role") TO "service_role";



GRANT ALL ON FUNCTION "public"."is_organization_admin"("org_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."is_organization_admin"("org_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."is_organization_admin"("org_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."is_organization_member"("org_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."is_organization_member"("org_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."is_organization_member"("org_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."manage_organization_secret"("org_id" "uuid", "secret_name" "text", "secret_value" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."manage_organization_secret"("org_id" "uuid", "secret_name" "text", "secret_value" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."manage_organization_secret"("org_id" "uuid", "secret_name" "text", "secret_value" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."remove_organization_member"("org_id" "uuid", "member_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."remove_organization_member"("org_id" "uuid", "member_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."remove_organization_member"("org_id" "uuid", "member_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."set_deployment_timeout"() TO "anon";
GRANT ALL ON FUNCTION "public"."set_deployment_timeout"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."set_deployment_timeout"() TO "service_role";



GRANT ALL ON FUNCTION "public"."update_job_status_if_in_progress"("_job_id" "text", "_new_status" "text", "_worker_id" "text", "_job_outputs" "jsonb", "_job_script" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."update_job_status_if_in_progress"("_job_id" "text", "_new_status" "text", "_worker_id" "text", "_job_outputs" "jsonb", "_job_script" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."update_job_status_if_in_progress"("_job_id" "text", "_new_status" "text", "_worker_id" "text", "_job_outputs" "jsonb", "_job_script" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."update_organization"("org_id" "uuid", "new_name" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."update_organization"("org_id" "uuid", "new_name" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."update_organization"("org_id" "uuid", "new_name" "text") TO "service_role";



GRANT ALL ON TABLE "public"."events" TO "anon";
GRANT ALL ON TABLE "public"."events" TO "authenticated";
GRANT ALL ON TABLE "public"."events" TO "service_role";



GRANT ALL ON SEQUENCE "public"."events_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."events_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."events_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."files" TO "anon";
GRANT ALL ON TABLE "public"."files" TO "authenticated";
GRANT ALL ON TABLE "public"."files" TO "service_role";



GRANT ALL ON TABLE "public"."organization_members" TO "anon";
GRANT ALL ON TABLE "public"."organization_members" TO "authenticated";
GRANT ALL ON TABLE "public"."organization_members" TO "service_role";



GRANT ALL ON TABLE "public"."organization_secrets" TO "anon";
GRANT ALL ON TABLE "public"."organization_secrets" TO "authenticated";
GRANT ALL ON TABLE "public"."organization_secrets" TO "service_role";



GRANT ALL ON TABLE "public"."organizations" TO "anon";
GRANT ALL ON TABLE "public"."organizations" TO "authenticated";
GRANT ALL ON TABLE "public"."organizations" TO "service_role";



GRANT ALL ON TABLE "public"."runs" TO "anon";
GRANT ALL ON TABLE "public"."runs" TO "authenticated";
GRANT ALL ON TABLE "public"."runs" TO "service_role";



GRANT ALL ON SEQUENCE "public"."runs_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."runs_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."runs_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."tokens" TO "anon";
GRANT ALL ON TABLE "public"."tokens" TO "authenticated";
GRANT ALL ON TABLE "public"."tokens" TO "service_role";



GRANT ALL ON TABLE "public"."worker" TO "anon";
GRANT ALL ON TABLE "public"."worker" TO "authenticated";
GRANT ALL ON TABLE "public"."worker" TO "service_role";



ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "service_role";






RESET ALL;
