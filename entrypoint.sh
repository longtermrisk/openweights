#!/bin/bash

echo "[$(date)] Starting entrypoint script"

# Add public keys to authorized_keys
echo "[$(date)] Checking for PUBLIC_KEY environment variable"
if [ -n "$PUBLIC_KEY" ]; then
    echo "[$(date)] Setting up SSH public key"
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    echo "[$(date)] SSH public key setup completed"
else
    echo "[$(date)] No PUBLIC_KEY provided, skipping SSH key setup"
fi

# Repository checkout if needed
echo "[$(date)] Checking for OW_COMMIT environment variable"
if [ -n "$OW_COMMIT" ]; then
    echo "[$(date)] Starting repository checkout"
    python3 openweights/worker/services/checkout.py
    if [ $? -ne 0 ]; then
        echo "[$(date)] Repository checkout failed"
        exit 1
    fi
else
    echo "[$(date)] No OW_COMMIT specified, skipping repository checkout"
fi

# Login to huggingface
echo "[$(date)] Attempting to login to Hugging Face"
python3 openweights/worker/services/hf_login.py
echo "[$(date)] Hugging Face login completed"

# Generate SSH host keys and start SSH service
echo "[$(date)] Setting up SSH service"
ssh-keygen -A
service ssh start
echo "[$(date)] SSH service started"

# Print sshd logs to stdout
tail -f /var/log/auth.log &

# Start background services
echo "[$(date)] Starting HTTP log server on port 10101"
mkdir logs
python3 openweights/worker/services/log_server.py &

# Start TTL monitoring service
echo "[$(date)] Starting TTL monitoring service"
python3 openweights/worker/services/ttl_monitor.py &

echo "[$(date)] All services started"

# Execute the main application or run in dev mode
if [ "$OW_DEV" = "true" ]; then
    echo "[$(date)] Starting in development mode"
    exec tail -f /dev/null
else
    echo "[$(date)] Starting main application"
    exec python3 openweights/worker/main.py > >(tee logs/main) 2> >(tee -a logs/main >&2)
fi
