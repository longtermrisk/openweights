# Dashboard frontend

Packaged dashboards served by `ow serve` load their public Supabase URL and anon
key from the backend's `/config.js` endpoint before starting React. Configure
`SUPABASE_URL` and `SUPABASE_ANON_KEY` on the server; no frontend build-time
configuration is required. The endpoint never returns service-role credentials.

For standalone Vite development, set `VITE_SUPABASE_URL` and
`VITE_SUPABASE_ANON_KEY` in `.env.local`, then run `npm ci` and `npm run dev`.
Runtime configuration takes precedence over these development settings.

Run `npm run build` to update the packaged backend static assets. CI builds
without Vite environment variables and runs `python tests/dashboard_browser_smoke.py`
from the repository root to verify the login page actually renders in Chromium.
The smoke test requires Playwright and its Chromium browser. Pass `--url URL`
to check an existing deployment.
