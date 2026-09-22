# OpenWeights 0.13.3

Create a token with a lifetime USD budget using
`ow token create --name experiment --spending-limit-usd 100`, or the dashboard's
new spending-limit field. The token and budget are saved atomically. Existing
API-key credentials authorized to create tokens can choose an initial budget;
editing existing budgets still requires a signed-in organization admin.

Apply `supabase/migrations/20260922180000_token_creation_limit.sql` before using
the new option. Worker images remain v0.13.1. Budget exhaustion continues to cancel
queued/running jobs for the key and reject new submissions; completed jobs stay
completed. Limits include allocated overhead and may be exceeded during shutdown.
