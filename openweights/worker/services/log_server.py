#!/usr/bin/env python3
"""
HTTP log server that serves log files on port 10101
"""
import http.server
import os
import socketserver
import sys


class LogHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # If path is /logs, serve logs/main
        if self.path == "/logs" or self.path == "/logs/":
            file_path = "logs/main"
        else:
            # Remove leading slash and ensure path is within logs directory
            path = self.path.lstrip("/")
            file_path = os.path.join("logs", path)

        # Check if file exists and is within logs directory
        if os.path.exists(file_path) and os.path.commonprefix(
            [os.path.abspath(file_path), os.path.abspath("logs")]
        ) == os.path.abspath("logs"):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File not found")


def start_server():
    """Start the log server on port 10101"""
    try:
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)

        with socketserver.TCPServer(("", 10101), LogHandler) as httpd:
            print("Starting HTTP log server on port 10101")
            httpd.serve_forever()
    except Exception as e:
        print(f"Failed to start log server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_server()
