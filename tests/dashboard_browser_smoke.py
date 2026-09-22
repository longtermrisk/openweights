"""Verify the built dashboard bootstraps in Chromium, without build-time secrets."""

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import sync_playwright


@contextmanager
def dashboard(url):
    if url:
        yield url
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    backend = Path(__file__).resolve().parents[1] / "openweights/dashboard/backend"
    process = subprocess.Popen(
        [
            os.getenv("OW_SERVER_PYTHON", sys.executable),
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=backend,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("Dashboard server exited before startup")
            try:
                with urllib.request.urlopen(url + "/config.js", timeout=1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("Dashboard server did not start")
        yield url
    finally:
        process.terminate()
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", help="Check an existing deployment instead of a local server"
    )
    args = parser.parse_args()
    with dashboard(args.url) as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.getenv("OW_CHROMIUM_PATH"),
            args=["--no-sandbox"],
        )
        try:
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            response = page.goto(url, wait_until="networkidle")
            assert response.ok
            page.get_by_label("Email Address").wait_for(state="visible")
            config = page.evaluate("window.__OPENWEIGHTS_CONFIG__")
            assert config and config["supabaseUrl"] and config["supabaseAnonKey"]
            assert not errors, errors
            print("Dashboard browser startup passed")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
