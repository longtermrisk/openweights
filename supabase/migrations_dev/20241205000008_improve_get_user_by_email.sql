-- Drop existing function
drop function if exists public.get_user_by_email(varchar);

-- Recreate function with parameter named 'email' and better error handling
create or replace function public.get_user_by_email(email varchar(255))
returns table (
    user_id uuid,
    user_email varchar(255)
) security definer
set search_path = public
language plpgsql
as $$
begin
    if email is null then
        raise exception 'Email parameter cannot be null';
    end if;

    return query
    select 
        au.id as user_id,
        au.email as user_email
    from auth.users au
    where lower(au.email) = lower(get_user_by_email.email)
    limit 1;

    if not found then
        raise exception 'User with email % not found', email;
    end if;
end;
$$;

-- Grant execute permissions on the function
grant execute on function public.get_user_by_email(varchar) to authenticated;