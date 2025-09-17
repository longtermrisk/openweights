# TTL (Time To Live) Feature

The TTL feature provides automatic pod termination to prevent runaway costs and ensure resource cleanup.

## Overview

- **Default TTL**: 24 hours for all pods
- **Automatic termination**: Pods self-terminate when TTL expires
- **Extensible**: TTL can be extended from within the pod
- **Dev mode support**: TTL monitoring runs for both dev and worker instances

## Usage

### Starting pods with custom TTL

```bash
# Start dev instance with default 24-hour TTL
python openweights/cluster/start_runpod.py A100 default --dev_mode=true

# Start dev instance with 2-hour TTL
python openweights/cluster/start_runpod.py A100 default --dev_mode=true --ttl_hours=2

# Start worker with 12-hour TTL
python openweights/cluster/start_runpod.py A100 finetuning --ttl_hours=12
```

### Managing TTL from within a pod

Once inside a pod, use the TTL manager utility:

```bash
# Check current TTL status
python openweights/worker/services/ttl_manager.py --check

# Extend TTL by 5 more hours
python openweights/worker/services/ttl_manager.py --extend 5

# Set TTL to 10 hours from now
python openweights/worker/services/ttl_manager.py --set 10
```

### Manual TTL management

You can also manually update the TTL by editing `~/shutdown.txt`:

```bash
python3 -c "
import datetime
with open('~/shutdown.txt', 'w') as f:
    new_time = datetime.datetime.now() + datetime.timedelta(hours=48)
    f.write(new_time.isoformat())
print(f'TTL extended to {new_time}')
"
```

## How it works

1. **TTL Setup**: When a pod starts, the TTL monitor service calculates the shutdown time and writes it to `~/shutdown.txt`
2. **Monitoring**: A background service checks the shutdown time every minute
3. **Termination**: When the current time exceeds the shutdown time, the service terminates the pod using the RunPod API
4. **Extension**: Jobs or users can extend the TTL by updating the shutdown time in the file

## Architecture

- **TTL Monitor Service**: `openweights/worker/services/ttl_monitor.py`
- **TTL Manager Utility**: `openweights/worker/services/ttl_manager.py`
- **Configuration**: TTL passed via `TTL_HOURS` environment variable
- **Shutdown File**: `~/shutdown.txt` contains ISO format datetime

## Environment Variables

- `TTL_HOURS`: Number of hours for TTL (default: 24)
- `RUNPOD_API_KEY`: RunPod API key for pod termination
- `OW_DEV`: Indicates if running in dev mode (affects other services, not TTL)

## Notes

- TTL monitoring runs for both dev and worker instances
- This provides an additional safety net especially for dev instances
- Pod ID is automatically detected from RunPod metadata API
- Failed termination attempts are retried every minute
- TTL can be reset/extended unlimited times before expiration
