#!/usr/bin/env python3
"""
Hugging Face login service
"""
import os
import sys

from huggingface_hub.hf_api import HfFolder


def login():
    """Login to Hugging Face using HF_TOKEN from environment"""
    try:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN not found in environment")
            return False

        HfFolder.save_token(hf_token)
        print("Successfully logged in to Hugging Face")
        return True
    except Exception as e:
        print(f"Failed to login to Hugging Face: {e}")
        return False


if __name__ == "__main__":
    success = login()
    sys.exit(0 if success else 1)
