-- Storage and files table policies

-- First, ensure the files bucket exists
INSERT INTO storage.buckets (id, name, public)
VALUES ('files', 'files', false)
ON CONFLICT (id) DO NOTHING;

-- Drop existing storage policies if they exist
DROP POLICY IF EXISTS "Organization members can upload files" ON storage.objects;
DROP POLICY IF EXISTS "Organization members can read files" ON storage.objects;
DROP POLICY IF EXISTS "Organization members can update files" ON storage.objects;
DROP POLICY IF EXISTS "Organization members can delete files" ON storage.objects;

-- Storage policies
CREATE POLICY "Organization members can upload files"
ON storage.objects
FOR INSERT
TO authenticated
WITH CHECK (
    bucket_id = 'files' AND
    public.is_organization_member(public.get_path_organization_id(name))
);

CREATE POLICY "Organization members can read files"
ON storage.objects
FOR SELECT
TO authenticated
USING (
    bucket_id = 'files' AND
    public.is_organization_member(public.get_path_organization_id(name))
);

CREATE POLICY "Organization members can update files"
ON storage.objects
FOR UPDATE
TO authenticated
USING (
    bucket_id = 'files' AND
    public.is_organization_member(public.get_path_organization_id(name))
)
WITH CHECK (
    bucket_id = 'files' AND
    public.is_organization_member(public.get_path_organization_id(name))
);

CREATE POLICY "Organization members can delete files"
ON storage.objects
FOR DELETE
TO authenticated
USING (
    bucket_id = 'files' AND
    public.is_organization_member(public.get_path_organization_id(name))
);

-- Fix files table RLS policies
DROP POLICY IF EXISTS "Organization members can access files" ON public.files;
DROP POLICY IF EXISTS "Organization members can read files" ON public.files;
DROP POLICY IF EXISTS "Organization members can insert files" ON public.files;
DROP POLICY IF EXISTS "Organization members can update files" ON public.files;
DROP POLICY IF EXISTS "Organization members can delete files" ON public.files;

-- Files table policies
CREATE POLICY "Organization members can read files"
    ON public.files FOR SELECT
    USING (public.is_organization_member(organization_id));

CREATE POLICY "Organization members can insert files"
    ON public.files FOR INSERT
    WITH CHECK (organization_id = public.get_organization_from_token());

CREATE POLICY "Organization members can update files"
    ON public.files FOR UPDATE
    USING (public.is_organization_member(organization_id))
    WITH CHECK (public.is_organization_member(organization_id));

CREATE POLICY "Organization members can delete files"
    ON public.files FOR DELETE
    USING (public.is_organization_member(organization_id));
