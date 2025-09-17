#!/usr/bin/env python3
"""
TTL monitoring service that terminates pods after expiration
"""
import datetime
import os
import sys
import time


def setup_ttl():
    """Setup initial TTL based on environment variable"""
    ttl_hours = float(os.environ.get("TTL_HOURS", "24"))
    shutdown_time = datetime.datetime.now() + datetime.timedelta(hours=ttl_hours)

    shutdown_file = os.path.expanduser("~/shutdown.txt")
    with open(shutdown_file, "w") as f:
        f.write(shutdown_time.isoformat())

    print(f"TTL set to {ttl_hours} hours. Shutdown scheduled for: {shutdown_time}")
    return shutdown_file


def get_pod_id():
    """Get the current pod ID from RunPod metadata or environment"""
    # First try environment variable
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        return pod_id

    # Try to get from RunPod metadata API
    try:
        import httpx

        with httpx.Client() as client:
            response = client.get(
                "http://metadata.runpod.ai/v1/instance/id", timeout=10
            )
            if response.status_code == 200:
                return response.text.strip()
    except Exception as e:
        print(f"Could not get pod ID from metadata API: {e}")

    return None


def terminate_pod():
    """Terminate the current pod using RunPod API"""
    try:
        import runpod

        api_key = os.environ.get("RUNPOD_API_KEY")
        pod_id = get_pod_id()

        if not api_key:
            print("ERROR: RUNPOD_API_KEY not found in environment")
            return False

        if not pod_id:
            print("ERROR: Could not determine pod ID")
            return False

        runpod.api_key = api_key
        result = runpod.terminate_pod(pod_id)
        print(f"Pod termination initiated for {pod_id}: {result}")
        return True

    except ImportError:
        print("ERROR: runpod package not available")
        return False
    except Exception as e:
        print(f"ERROR: Failed to terminate pod: {e}")
        return False


def monitor_ttl():
    """Monitor TTL and terminate pod when expired"""
    shutdown_file = setup_ttl()

    print("Starting TTL monitoring service...")

    while True:
        try:
            if os.path.exists(shutdown_file):
                with open(shutdown_file, "r") as f:
                    shutdown_time_str = f.read().strip()

                try:
                    shutdown_time = datetime.datetime.fromisoformat(shutdown_time_str)
                    current_time = datetime.datetime.now()

                    if current_time >= shutdown_time:
                        print(
                            f"TTL expired at {shutdown_time}. Current time: {current_time}"
                        )
                        print("Initiating pod termination...")

                        if terminate_pod():
                            print("Pod termination successful")
                            break
                        else:
                            print("Pod termination failed, will retry in 60 seconds")
                    else:
                        time_left = shutdown_time - current_time
                        print(f"TTL check: {time_left} remaining until shutdown")

                except ValueError as e:
                    print(f"Invalid shutdown time format in {shutdown_file}: {e}")
                    # Re-setup TTL if file is corrupted
                    shutdown_file = setup_ttl()
            else:
                print(f"Shutdown file {shutdown_file} not found, recreating...")
                shutdown_file = setup_ttl()

        except Exception as e:
            print(f"Error in TTL monitoring: {e}")

        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    try:
        monitor_ttl()
    except KeyboardInterrupt:
        print("TTL monitoring service stopped")
        sys.exit(0)
    except Exception as e:
        print(f"TTL monitoring service failed: {e}")
        sys.exit(1)
