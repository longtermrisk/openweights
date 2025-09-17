#!/bin/bash

# Configuration
WORK_DIR="/workspace/openweights"
DASHBOARD_DIR="$WORK_DIR/openweights/dashboard"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# # Update code
# log "Pulling latest code..."
# cd "$WORK_DIR"
# git pull origin main

# Build frontend
log "Building frontend..."
cd $DASHBOARD_DIR/frontend
npm run build

# Update backend static files
log "Updating backend static files..."
rm -rf ../backend/static/*
cp -r dist/* ../backend/static/

# Restart backend service
log "Restarting backend service..."
cd ../backend
if [ -f "backend.pid" ]; then
    if ps -p $(cat backend.pid) > /dev/null; then
        kill $(cat backend.pid)
        sleep 2
    fi
fi
export SITE_URL=https://kzy2zyhynxvjz7-8124.proxy.runpod.net
export ADDITIONAL_REDIRECT_URLS=https://kzy2zyhynxvjz7-8124.proxy.runpod.net/**
export API_EXTERNAL_URL=https://kzy2zyhynxvjz7-8124.proxy.runpod.net
nohup uvicorn main:app --port 8124 --host 0.0.0.0 --workers 10  > backend.log 2>&1 & echo $! > backend.pid

log "Deployment completed!"
