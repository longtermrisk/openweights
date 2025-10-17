#!/usr/bin/env bash

# Usage: source use.sh dev
# Copies .env.dev → .env and supabase/config.toml.dev → supabase/config.toml
# Loads environment variables into current shell session

if [ -z "$1" ]; then
  echo "Usage: source use.sh <env_name>"
  return 1
fi

ENV_NAME="$1"
ENV_FILE=".env.$ENV_NAME"
SUPABASE_FILE="supabase/config.toml.$ENV_NAME"
SUPABASE_DEFAULT="supabase/config.toml.dev"

# --- Handle .env ---
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: $ENV_FILE does not exist."
  return 1
fi

cp "$ENV_FILE" .env

# --- Handle supabase config ---
if [ -f "$SUPABASE_FILE" ]; then
  cp "$SUPABASE_FILE" supabase/config.toml
  echo "Using Supabase config: $SUPABASE_FILE"
else
  cp "$SUPABASE_DEFAULT" supabase/config.toml
  echo "Warning: $SUPABASE_FILE not found, using default: $SUPABASE_DEFAULT"
fi

# --- Load env vars into current shell ---
set -o allexport
source .env
set +o allexport

echo "✅ Switched to environment: $ENV_NAME"
