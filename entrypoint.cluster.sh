#!/bin/bash

echo "[$(date)] Starting cluster/dashboard entrypoint script"

# Determine which command to run based on OW_CMD environment variable
# Options: "cluster", "serve", or "both" (default)
OW_CMD="${OW_CMD:-both}"

echo "[$(date)] OW_CMD set to: $OW_CMD"

# Start the dashboard backend if requested
if [ "$OW_CMD" = "serve" ] || [ "$OW_CMD" = "both" ]; then
    echo "[$(date)] Starting dashboard backend"

    # Set default environment variables if not provided
    export SITE_URL="${SITE_URL:-http://localhost:8124}"
    export API_EXTERNAL_URL="${API_EXTERNAL_URL:-http://localhost:8124}"

    # Start dashboard in background if running both, otherwise in foreground
    if [ "$OW_CMD" = "both" ]; then
        mkdir -p /openweights/logs
        ow serve > /openweights/logs/dashboard.log 2>&1 &
        DASHBOARD_PID=$!
        echo "[$(date)] Dashboard backend started with PID: $DASHBOARD_PID"
    else
        echo "[$(date)] Starting dashboard backend (foreground mode)"
        exec ow serve
    fi
fi

# Start the cluster manager if requested
if [ "$OW_CMD" = "cluster" ] || [ "$OW_CMD" = "both" ]; then
    echo "[$(date)] Starting cluster manager"
    mkdir -p /openweights/logs

    # Build cluster command with optional flags
    CLUSTER_CMD="ow cluster"
    if [ -n "$OW_CLUSTER_FLAGS" ]; then
        CLUSTER_CMD="$CLUSTER_CMD $OW_CLUSTER_FLAGS"
        echo "[$(date)] Cluster flags: $OW_CLUSTER_FLAGS"
    fi

    if [ "$OW_CMD" = "both" ]; then
        # Run cluster manager in foreground when running both
        exec $CLUSTER_CMD > >(tee /openweights/logs/cluster.log) 2> >(tee -a /openweights/logs/cluster.log >&2)
    else
        # Run cluster manager in foreground when running only cluster
        exec $CLUSTER_CMD > >(tee /openweights/logs/cluster.log) 2> >(tee -a /openweights/logs/cluster.log >&2)
    fi
fi

# If we get here, no valid command was specified
echo "[$(date)] Error: Invalid OW_CMD value: $OW_CMD"
echo "[$(date)] Valid options: cluster, serve, both"
exit 1
