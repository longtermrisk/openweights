-- Update the get_jwt_secret function to use the vault
create or replace function get_jwt_secret()
returns text
language plpgsql
security definer
as $$
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
