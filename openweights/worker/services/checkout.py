#!/usr/bin/env python3
"""
Repository checkout service for specific commits
"""
import os
import shutil
import subprocess
import sys


def checkout_commit():
    """Checkout specific commit if OW_COMMIT is set"""
    commit = os.environ.get("OW_COMMIT")
    if not commit:
        print("No OW_COMMIT specified, skipping repository checkout")
        return True

    try:
        print(f"Starting repository checkout for commit: {commit}")

        # Remove existing openweights directory
        if os.path.exists("openweights"):
            shutil.rmtree("openweights")

        # Clone repository
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/longtermrisk/openweights.git",
                "openweights_dev",
            ],
            check=True,
        )

        # Checkout specific commit
        os.chdir("openweights_dev")
        subprocess.run(["git", "checkout", commit], check=True)

        # Move openweights directory back
        os.chdir("..")
        shutil.move("openweights_dev/openweights", "openweights")
        shutil.rmtree("openweights_dev")

        print("Repository checkout completed")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed to checkout repository: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during checkout: {e}")
        return False


if __name__ == "__main__":
    success = checkout_commit()
    sys.exit(0 if success else 1)
