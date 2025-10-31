-- Fix files INSERT policy to work with API tokens

DROP POLICY IF EXISTS "Organization members can insert files" ON public.files;

-- Allow authenticated users to insert files if they're a member of the organization
CREATE POLICY "Organization members can insert files"
    ON public.files FOR INSERT
    WITH CHECK (public.is_organization_member(organization_id));
