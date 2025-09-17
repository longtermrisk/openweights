-- Ensure 'files' bucket exists in the shadow DB used by `supabase db pull`.
-- This runs before migrations during local/shadow setup and is a NOOP on subsequent runs.

insert into storage.buckets (id, name, public)
values ('files', 'files', false)
on conflict (id) do nothing;
