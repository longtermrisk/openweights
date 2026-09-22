# OpenWeights 0.13.2

Fixes the blank dashboard caused by missing frontend Supabase environment
variables in packaged builds. The dashboard now reads its public configuration
from the backend at startup, so releases do not require build-time credentials.

Adds API coverage for public configuration and a Chromium startup check in CI.
No database migration is required. Worker images remain at v0.13.1 because this
patch only changes the dashboard.
